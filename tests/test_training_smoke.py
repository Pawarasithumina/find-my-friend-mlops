
import math

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    Subset,
)

from src.models.architectures import create_model
from src.data.dataset_loader import FaceOpsDataset
from src.training.engine import fit_model


torch.manual_seed(42)
torch.set_num_threads(2)


def balanced_subset(dataset, samples_per_class):
    indices = []

    for class_id in range(4):
        available = (
            (dataset.labels == class_id)
            .nonzero(as_tuple=True)[0]
            .tolist()
        )

        if len(available) < samples_per_class:
            raise ValueError(
                f"Not enough images for class {class_id}"
            )

        indices.extend(
            available[:samples_per_class]
        )

    return Subset(dataset, indices)


def main():
    print("\n===== FACEOPS TRAINING SMOKE TEST =====")

    train_dataset = FaceOpsDataset(
        split="train",
        model_name="custom_cnn",
        augment=True,
    )

    val_dataset = FaceOpsDataset(
        split="val",
        model_name="custom_cnn",
        augment=False,
    )

    train_subset = balanced_subset(
        train_dataset,
        samples_per_class=4,
    )

    val_subset = balanced_subset(
        val_dataset,
        samples_per_class=2,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    model = create_model(
        model_name="custom_cnn",
        num_classes=4,
        pretrained=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    criterion = nn.CrossEntropyLoss()

    result = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=2,
        patience=2,
        device="cpu",
    )

    history = result["history"]

    assert len(history) == 2
    assert result["best_epoch"] in (1, 2)

    for record in history:
        for name in (
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy",
            "train_macro_f1",
            "val_macro_f1",
        ):
            assert math.isfinite(record[name])

        assert 0 <= record["val_accuracy"] <= 1
        assert 0 <= record["val_macro_f1"] <= 1

    print("\nTraining samples:", len(train_subset))
    print("Validation samples:", len(val_subset))
    print("History records:", len(history))
    print("Training engine: PASSED")

    print("\n===== ALL SMOKE TESTS PASSED =====")


if __name__ == "__main__":
    main()
