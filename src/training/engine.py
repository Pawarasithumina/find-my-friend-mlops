
import copy

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


NUM_CLASSES = 4


# ============================================
# TRAIN / VALIDATION EPOCH
# ============================================

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

    if total_samples == 0:
        raise ValueError("DataLoader is empty")

    metrics = {
        "loss": total_loss / total_samples,

        "accuracy": accuracy_score(
            all_labels,
            all_predictions,
        ),

        "macro_f1": f1_score(
            all_labels,
            all_predictions,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        ),
    }

    return metrics


# ============================================
# COMPLETE TRAINING PROCEDURE
# ============================================

def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs=30,
    patience=7,
    device="cpu",
    on_epoch=None,
):
    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    if patience < 1:
        raise ValueError("patience must be >= 1")

    device = torch.device(device)
    model = model.to(device)

    history = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None

    epochs_without_improvement = 0

    print("\n===== FACEOPS MODEL TRAINING =====")
    print("Device:", device)
    print("Maximum epochs:", epochs)
    print("Early stopping patience:", patience)

    for epoch in range(1, epochs + 1):

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        result = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],

            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }

        history.append(result)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train Loss: {result['train_loss']:.4f} | "
            f"Val Loss: {result['val_loss']:.4f} | "
            f"Val Acc: {result['val_accuracy']:.4f} | "
            f"Val Macro-F1: {result['val_macro_f1']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        # Future integration point for
        # MLflow logging and Optuna pruning.
        if on_epoch is not None:
            on_epoch(result)

        if epochs_without_improvement >= patience:
            print("\nEarly stopping triggered.")
            break

    # Restore the best validation checkpoint.
    if best_weights is not None:
        model.load_state_dict(best_weights)

    print("\n===== TRAINING COMPLETED =====")
    print("Best epoch:", best_epoch)
    print(f"Best validation loss: {best_val_loss:.4f}")

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
