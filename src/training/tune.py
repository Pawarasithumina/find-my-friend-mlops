
import argparse
import random

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from src.models.architectures import create_model
from src.data.dataset_loader import FaceOpsDataset
from src.training.engine import fit_model


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_subset(dataset, per_class):
    indices = []

    for class_id in range(4):
        available = (
            (dataset.labels == class_id)
            .nonzero(as_tuple=True)[0]
            .tolist()
        )

        indices.extend(available[:per_class])

    return Subset(dataset, indices)


def make_loaders(model_name, batch_size, smoke, seed):

    train = FaceOpsDataset(
        "train", model_name, augment=True
    )

    val = FaceOpsDataset(
        "val", model_name, augment=False
    )

    if smoke:
        train = balanced_subset(train, 4)
        val = balanced_subset(val, 2)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0
    )

    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


def log_loss_curve(history):

    epochs = [x["epoch"] for x in history]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        epochs,
        [x["train_loss"] for x in history],
        label="Training loss"
    )

    ax.plot(
        epochs,
        [x["val_loss"] for x in history],
        label="Validation loss"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("FaceOps Optuna Trial")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    mlflow.log_figure(
        fig,
        "training/loss_curve.png"
    )

    plt.close(fig)


def run_study(model_name, trials, epochs, smoke):

    set_seed(SEED)
    torch.set_num_threads(2)

    tracking_uri = (
        "http://127.0.0.1:5050"
    )

    mlflow.set_tracking_uri(tracking_uri)

    experiment = (
        "FaceOps_Optuna_Smoke"
        if smoke else f"FaceOps_Optuna_{model_name}"
    )

    mlflow.set_experiment(experiment)

    sampler = optuna.samplers.TPESampler(
        seed=SEED
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=2
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner
    )

    def objective(trial):

        lr = trial.suggest_float(
            "learning_rate",
            1e-4,
            3e-3,
            log=True
        )

        dropout = trial.suggest_float(
            "dropout",
            0.2,
            0.5
        )

        batch_size = trial.suggest_categorical(
            "batch_size",
            [8, 16]
        )

        weight_decay = trial.suggest_float(
            "weight_decay",
            1e-6,
            1e-3,
            log=True
        )

        trial_seed = SEED + trial.number
        set_seed(trial_seed)

        with mlflow.start_run(
            run_name=f"{model_name}_trial_{trial.number}",
            nested=True
        ) as run:

            trial.set_user_attr(
                "mlflow_run_id",
                run.info.run_id
            )

            mlflow.log_params({
                **trial.params,
                "model_name": model_name,
                "dataset_version": 2,
                "epochs": epochs,
                "seed": trial_seed,
                "pretrained": not smoke and (
                    model_name != "custom_cnn"
                ),
                "freeze_backbone": (
                    model_name != "custom_cnn"
                )
            })

            mlflow.set_tags({
                "project": "FaceOps_TwinTrack",
                "run_type": (
                    "optuna_smoke"
                    if smoke else "optuna_trial"
                )
            })

            train_loader, val_loader = make_loaders(
                model_name,
                batch_size,
                smoke,
                trial_seed
            )

            model = create_model(
                model_name,
                num_classes=4,
                dropout=dropout,
                pretrained=(
                    not smoke
                    and model_name != "custom_cnn"
                ),
                freeze_backbone=(
                    model_name != "custom_cnn"
                )
            )

            optimizer = torch.optim.AdamW(
                filter(
                    lambda p: p.requires_grad,
                    model.parameters()
                ),
                lr=lr,
                weight_decay=weight_decay
            )

            criterion = nn.CrossEntropyLoss()

            def epoch_callback(metrics):
                step = metrics["epoch"]

                mlflow.log_metrics(
                    {
                        key: float(value)
                        for key, value in metrics.items()
                        if key != "epoch"
                    },
                    step=step
                )

                trial.report(
                    float(metrics["val_loss"]),
                    step=step
                )

                if trial.should_prune():
                    raise optuna.TrialPruned()

            result = fit_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                epochs=epochs,
                patience=min(5, epochs),
                device=DEVICE,
                on_epoch=epoch_callback
            )

            mlflow.log_metrics({
                "best_val_loss": result["best_val_loss"],
                "best_epoch": result["best_epoch"]
            })

            mlflow.log_dict(
                {"history": result["history"]},
                "training/history.json"
            )

            log_loss_curve(result["history"])

            return float(result["best_val_loss"])

    with mlflow.start_run(
        run_name=f"{model_name}_optuna_study"
    ):

        mlflow.log_params({
            "model_name": model_name,
            "number_of_trials": trials,
            "max_epochs": epochs,
            "dataset_version": 2,
            "smoke_test": smoke,
            "objective": "minimize_validation_loss"
        })

        study.optimize(
            objective,
            n_trials=trials,
            gc_after_trial=True
        )

        mlflow.log_metric(
            "best_validation_loss",
            study.best_value
        )

        mlflow.log_params({
            f"best_{key}": value
            for key, value in study.best_params.items()
        })

        mlflow.set_tag(
            "best_trial_run_id",
            study.best_trial.user_attrs[
                "mlflow_run_id"
            ]
        )

    print("\n===== OPTUNA STUDY COMPLETED =====")
    print("Model:", model_name)
    print("Best trial:", study.best_trial.number)
    print("Best validation loss:", study.best_value)
    print("Best hyperparameters:")

    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "custom_cnn",
            "mobilenet_v3",
            "resnet18"
        ],
        default="custom_cnn"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=8
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    args = parser.parse_args()

    run_study(
        model_name=args.model,
        trials=2 if args.smoke else args.trials,
        epochs=2 if args.smoke else args.epochs,
        smoke=args.smoke
    )
