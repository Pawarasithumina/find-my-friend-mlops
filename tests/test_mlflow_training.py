
import os
import random

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn

from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader

from src.models.architectures import create_model
from src.data.dataset_loader import FaceOpsDataset
from src.training.engine import fit_model
from tests.test_training_smoke import balanced_subset


# ============================================
# CONFIGURATION
# ============================================

SEED = 42
MODEL_NAME = "custom_cnn"
DATASET_VERSION = 2

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5050"
)

EXPERIMENT_NAME = "FaceOps_Training_Smoke"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)


# ============================================
# CREATE DATA LOADERS
# ============================================

def prepare_loaders():

    train_data = FaceOpsDataset(
        split="train",
        model_name=MODEL_NAME,
        augment=True
    )

    val_data = FaceOpsDataset(
        split="val",
        model_name=MODEL_NAME,
        augment=False
    )

    train_subset = balanced_subset(
        train_data,
        samples_per_class=4
    )

    val_subset = balanced_subset(
        val_data,
        samples_per_class=2
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_subset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


# ============================================
# MLFLOW EPOCH CALLBACK
# ============================================

def log_epoch(metrics):

    epoch = metrics["epoch"]

    mlflow.log_metrics(
        {
            name: float(value)
            for name, value in metrics.items()
            if name != "epoch"
        },
        step=epoch
    )


# ============================================
# CREATE LOSS CURVE
# ============================================

def log_loss_curve(history):

    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        epochs,
        train_loss,
        marker="o",
        label="Training Loss"
    )

    ax.plot(
        epochs,
        val_loss,
        marker="o",
        label="Validation Loss"
    )

    ax.set_title("FaceOps Training Smoke Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    mlflow.log_figure(
        fig,
        "training/loss_curve.png"
    )

    plt.close(fig)


# ============================================
# RUN MLFLOW TRAINING TEST
# ============================================

def main():

    print("\n===== FACEOPS MLFLOW TRAINING TEST =====")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_loader, val_loader = prepare_loaders()

    model = create_model(
        MODEL_NAME,
        num_classes=4,
        pretrained=False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run(
        run_name="custom_cnn_training_smoke"
    ) as run:

        run_id = run.info.run_id

        mlflow.log_params({
            "model_name": MODEL_NAME,
            "dataset_version": DATASET_VERSION,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "epochs": 2,
            "patience": 2,
            "train_samples": 16,
            "val_samples": 8,
            "seed": SEED
        })

        mlflow.set_tags({
            "project": "FaceOps_TwinTrack",
            "run_type": "smoke_test",
            "dataset_strategy":
                "chronological_known_grouped_unknown"
        })

        result = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=2,
            patience=2,
            device="cpu",
            on_epoch=log_epoch
        )

        history = result["history"]

        # Log the selected checkpoint metrics.
        mlflow.log_metrics({
            "best_epoch": result["best_epoch"],
            "best_val_loss": result["best_val_loss"]
        })

        # Store complete training history.
        mlflow.log_dict(
            {
                "history": history,
                "best_epoch": result["best_epoch"]
            },
            "training/history.json"
        )

        # Store loss curve in RustFS via MLflow.
        log_loss_curve(history)

        print("\nMLflow Run ID:", run_id)

    # ========================================
    # VERIFY MLFLOW RECORDS
    # ========================================

    client = MlflowClient(
        tracking_uri=TRACKING_URI
    )

    saved_run = client.get_run(run_id)

    assert saved_run.data.params["model_name"] == (
        MODEL_NAME
    )

    assert "val_loss" in saved_run.data.metrics
    assert "train_loss" in saved_run.data.metrics

    artifacts = client.list_artifacts(
        run_id,
        path="training"
    )

    artifact_names = {
        item.path for item in artifacts
    }

    assert "training/history.json" in artifact_names
    assert "training/loss_curve.png" in artifact_names

    print("\nMLflow metrics logging: PASSED")
    print("MLflow artifact logging: PASSED")
    print("Experiment:", EXPERIMENT_NAME)
    print("Tracking URI:", TRACKING_URI)

    print("\n===== ALL MLFLOW TESTS PASSED =====")


if __name__ == "__main__":
    main()
