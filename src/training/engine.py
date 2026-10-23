"""Shared supervised training engine with frozen-backbone BatchNorm handling."""
import copy
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score


NUM_CLASSES = 4


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training:
        # Frozen pretrained backbones should not update BatchNorm running
        # statistics on our small, single-session dataset.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                local_params = list(module.parameters(recurse=False))
                if local_params and all(not p.requires_grad for p in local_params):
                    module.eval()

    total_loss, total_samples = 0.0, 0
    labels_all, preds_all = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        n = labels.size(0)
        total_loss += loss.item() * n
        total_samples += n
        labels_all.extend(labels.detach().cpu().tolist())
        preds_all.extend(logits.argmax(dim=1).detach().cpu().tolist())

    if total_samples == 0:
        raise ValueError("DataLoader is empty")
    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(labels_all, preds_all),
        "macro_f1": f1_score(labels_all, preds_all,
                             labels=list(range(NUM_CLASSES)),
                             average="macro", zero_division=0),
    }


def fit_model(model, train_loader, val_loader, optimizer, criterion,
              epochs=30, patience=7, device="cpu", on_epoch=None):
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    device = torch.device(device)
    model = model.to(device)
    history, best_epoch = [], 0
    best_val_loss, best_weights = float("inf"), None
    unimproved = 0
    print("\n===== FACEOPS MODEL TRAINING =====")
    print("Device:", device)
    print("Maximum epochs:", epochs)
    print("Early stopping patience:", patience)

    for epoch in range(1, epochs + 1):
        train = run_epoch(model, train_loader, criterion, device, optimizer)
        val = run_epoch(model, val_loader, criterion, device)
        record = {"epoch": epoch}
        record.update({f"train_{k}": v for k, v in train.items()})
        record.update({f"val_{k}": v for k, v in val.items()})
        history.append(record)
        print(f"Epoch {epoch:03d}/{epochs:03d} | "
              f"Train Loss: {train['loss']:.4f} | "
              f"Val Loss: {val['loss']:.4f} | "
              f"Val Acc: {val['accuracy']:.4f} | "
              f"Val Macro-F1: {val['macro_f1']:.4f}")
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            unimproved = 0
        else:
            unimproved += 1
        if on_epoch is not None:
            on_epoch(record)
        if unimproved >= patience:
            print("Early stopping triggered.")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    print("\n===== TRAINING COMPLETED =====")
    print("Best epoch:", best_epoch)
    print(f"Best validation loss: {best_val_loss:.4f}")
    return {"model": model, "history": history,
            "best_epoch": best_epoch, "best_val_loss": best_val_loss}
