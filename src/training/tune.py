"""FaceOps v2: resumable Optuna tuning, MLflow metrics, and model checkpoints.

Runs locally, with MLflow served at http://127.0.0.1:5050.
The held-out test split is intentionally never loaded in this module.
"""

import argparse
import gc
import json
import os
import random
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset_loader import CLASS_NAMES, FaceOpsDataset
from src.models.architectures import create_model
from src.training.engine import fit_model


ROOT = Path(__file__).resolve().parents[2]
DATASET_VERSION = 2
BASE_SEED = 42
MODELS = ("custom_cnn", "mobilenet_v3", "resnet18")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
STORAGE_DIR = ROOT / "data" / "optuna"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_subset(dataset, n_per_class: int):
    indices = []
    for class_id in range(len(CLASS_NAMES)):
        found = (dataset.labels == class_id).nonzero(as_tuple=True)[0].tolist()
        if len(found) < n_per_class:
            raise ValueError(f"Class {class_id} has only {len(found)} samples")
        indices.extend(found[:n_per_class])
    return Subset(dataset, indices)


def build_loaders(model_name: str, batch_size: int, smoke: bool, seed: int):
    train = FaceOpsDataset("train", model_name, augment=True)
    val = FaceOpsDataset("val", model_name, augment=False)
    if smoke:
        train = sample_subset(train, 4)
        val = sample_subset(val, 2)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=generator, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def log_curves(history):
    xs = [x["epoch"] for x in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for prefix, title, ax in (
        ("loss", "Cross-entropy loss", axes[0]),
        ("macro_f1", "Macro-F1", axes[1]),
    ):
        ax.plot(xs, [x[f"train_{prefix}"] for x in history], label="Train")
        ax.plot(xs, [x[f"val_{prefix}"] for x in history], label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    mlflow.log_figure(fig, "training/curves.png")
    plt.close(fig)


def run_study(model_name: str, trials: int, epochs: int, smoke: bool = False):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    if trials < 1 or epochs < 1:
        raise ValueError("--trials and --epochs must be >= 1")

    # Keep the existing, verified tiny smoke-test behavior.
    if smoke:
        trials, epochs = 2, 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    set_seed(BASE_SEED)

    # SQLite is sufficient for sequential local studies; keep DB outside Git.
    # Do not run multiple writers against this SQLite study simultaneously.
    db_path = STORAGE_DIR / "faceops_optuna.sqlite3"
    storage_url = "sqlite:///" + str(db_path)
    variant = "smoke" if smoke else "production"
    study_name = f"faceops_v{DATASET_VERSION}_{model_name}_{variant}_epochs{epochs}"

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(
        "FaceOps_Optuna_Smoke_v2" if smoke else f"FaceOps_Production_{model_name}"
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=BASE_SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=3
        ),
    )

    print(f"\nStudy: {study_name}")
    print(f"Device: {device}; trials to ADD: {trials}; max epochs: {epochs}")
    print("The test split will NOT be used.")
    print(f"Prior trials: {len(study.trials)}")

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [8, 16])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        trial_seed = BASE_SEED + trial.number
        set_seed(trial_seed)

        pretrained = model_name != "custom_cnn" and not smoke
        freeze_backbone = model_name != "custom_cnn"

        with mlflow.start_run(
            run_name=f"{model_name}_trial_{trial.number}", nested=True
        ) as run:
            trial.set_user_attr("mlflow_run_id", run.info.run_id)
            mlflow.log_params({
                **trial.params,
                "model_name": model_name,
                "dataset_version": DATASET_VERSION,
                "dataset_split": "chronological_known_grouped_unknown",
                "epochs": epochs,
                "seed": trial_seed,
                "pretrained": pretrained,
                "freeze_backbone": freeze_backbone,
                "train_mode": variant,
            })
            mlflow.set_tags({
                "project": "FaceOps_TwinTrack",
                "dataset_limit": "known_classes_single_capture_session",
                "test_split_used": "false",
                "run_type": f"optuna_{variant}",
            })

            train_loader, val_loader = build_loaders(
                model_name, batch_size, smoke, trial_seed
            )
            model = create_model(
                model_name, num_classes=4, dropout=dropout,
                pretrained=pretrained, freeze_backbone=freeze_backbone,
            )
            trainable_parameters = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            mlflow.log_param("trainable_parameters", trainable_parameters)
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=learning_rate, weight_decay=weight_decay,
            )

            def epoch_callback(metrics):
                epoch = int(metrics["epoch"])
                mlflow.log_metrics(
                    {k: float(v) for k, v in metrics.items() if k != "epoch"},
                    step=epoch,
                )
                trial.report(float(metrics["val_loss"]), step=epoch)
                if not smoke and trial.should_prune():
                    raise optuna.TrialPruned()

            result = fit_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=nn.CrossEntropyLoss(),
                epochs=epochs,
                patience=min(6, epochs),
                device=device,
                on_epoch=epoch_callback,
            )
            mlflow.log_metrics({
                "best_val_loss": result["best_val_loss"],
                "best_epoch": result["best_epoch"],
            })
            mlflow.log_dict({
                "history": result["history"],
                "best_epoch": result["best_epoch"],
                "test_evaluated": False,
            }, "training/history.json")
            log_curves(result["history"])

            # Save CPU-only, reconstructable state_dict to MLflow's RustFS storage.
            # Later, an approved version will be packaged for MLflow Model Registry.
            with tempfile.TemporaryDirectory() as temp:
                checkpoint = Path(temp) / "best_state.pt"
                torch.save({
                    "state_dict": {
                        k: v.detach().cpu() for k, v in result["model"].state_dict().items()
                    },
                    "architecture": model_name,
                    "classes": CLASS_NAMES,
                    "dataset_version": DATASET_VERSION,
                    "training_mode": variant,
                    "hyperparameters": dict(trial.params),
                    "best_epoch": result["best_epoch"],
                    "best_val_loss": result["best_val_loss"],
                    "pretrained": pretrained,
                    "freeze_backbone": freeze_backbone,
                    "input_shape": [3, 224, 224],
                }, checkpoint)
                mlflow.log_artifact(str(checkpoint), artifact_path="checkpoint")

            val_loss = float(result["best_val_loss"])
            # Free model references on multi-trial GPU runs.
            del model, optimizer, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return val_loss

    with mlflow.start_run(run_name=f"{model_name}_{variant}_study"):
        mlflow.log_params({
            "model_name": model_name,
            "requested_additional_trials": trials,
            "max_epochs": epochs,
            "dataset_version": DATASET_VERSION,
            "study_name": study_name,
            "evaluation_split": "validation_only",
            "training_mode": variant,
        })
        study.optimize(objective, n_trials=trials, gc_after_trial=True)
        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            raise RuntimeError("No trials completed. Inspect logs before continuing.")
        best = study.best_trial
        mlflow.log_metric("best_val_loss", float(best.value))
        mlflow.log_metric("total_completed_trials", len(completed))
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.set_tag("best_trial_run_id", best.user_attrs["mlflow_run_id"])
        mlflow.log_dict({
            "study_name": study_name,
            "best_trial": best.number,
            "best_run_id": best.user_attrs["mlflow_run_id"],
            "best_validation_loss": best.value,
            "best_hyperparameters": best.params,
            "total_completed_trials": len(completed),
            "selection": "minimum_validation_loss",
            "test_set_used": False,
        }, "study/best_trial.json")

    print("\n===== FACEOPS OPTUNA STUDY COMPLETE =====")
    print("Model:", model_name)
    print("Study:", study_name)
    print("Best trial:", best.number)
    print("Best validation loss:", best.value)
    print("Best MLflow run ID:", best.user_attrs["mlflow_run_id"])
    print("Best hyperparameters:", json.dumps(best.params, indent=2))
    return {
        "model": model_name, "study_name": study_name,
        "best_trial": best.number, "best_run_id": best.user_attrs["mlflow_run_id"],
        "best_val_loss": float(best.value), "best_params": best.params,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="custom_cnn")
    parser.add_argument("--trials", type=int, default=6,
                        help="ADDITIONAL trials to run (study is resumable)")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_study(args.model, args.trials, args.epochs, args.smoke)
