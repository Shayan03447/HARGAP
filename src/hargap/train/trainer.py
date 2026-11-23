"""Shared training loop, loss calculation, and fold evaluation."""

import random
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, accuracy_score

from hargap.train.early_stopping import EarlyStopper
from hargap.train.budget import BudgetTracker


def set_seed(python_seed: int = 1, numpy_seed: int = 11, torch_seed: int = 111):
    """Sets deterministic random seeds across all libraries."""
    random.seed(python_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_class_weights(train_labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    Computes inverse frequency class weights from training labels only.
    Renormalized so that mean weight is 1.0.
    """
    counts = np.bincount(train_labels, minlength=n_classes).astype(np.float32)
    # Avoid zero division
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    normalized_weights = inv_freq / np.mean(inv_freq)
    return torch.tensor(normalized_weights, dtype=torch.float32)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, float, float, float, np.ndarray]:
    """
    Evaluates model on a DataLoader.
    Returns:
        all_preds: (N,) int
        all_targets: (N,) int
        macro_f1: float
        balanced_acc: float
        accuracy: float
        cm: (K, K) int array
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())

    y_pred = np.array(all_preds, dtype=np.int64)
    y_true = np.array(all_targets, dtype=np.int64)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)

    return y_pred, y_true, macro_f1, bal_acc, acc, cm


def train_fold(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    n_classes: int,
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs_max: int = 150,
    patience: int = 10,
    seeds: Dict[str, int] = {"python": 1, "numpy": 11, "torch": 111},
    use_amp: bool = False,
    device: Optional[torch.device] = None,
    budget_tracker: Optional[BudgetTracker] = None,
) -> Dict[str, Any]:
    """
    Executes the shared training procedure on a single fold.
    Returns:
        dictionary containing final test metrics, best validation score,
        epochs trained, parameter counts, and timing audit.
    """
    set_seed(seeds.get("python", 1), seeds.get("numpy", 11), seeds.get("torch", 111))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Class-weighted loss computed strictly from training labels
    class_weights = compute_class_weights(train_y, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )

    use_cuda_amp = bool(use_amp and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

    # Convert to PyTorch DataLoaders
    # Training loader has shuffle=True with seeded generator; Val/Test shuffle=False
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_gen = torch.Generator()
    train_gen.manual_seed(seeds.get("torch", 111))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=train_gen, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    stopper = EarlyStopper(patience=patience, max_epochs=epochs_max)
    if budget_tracker:
        budget_tracker.start_fold()

    history = []

    for epoch in range(1, epochs_max + 1):
        model.train()
        total_loss = 0.0

        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()

            autocast_enabled = bool(use_amp and device.type == "cuda")
            try:
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=autocast_enabled)
            except Exception:
                autocast_ctx = torch.cuda.amp.autocast(enabled=autocast_enabled)
            with autocast_ctx:
                logits = model(b_x)
                loss = criterion(logits, b_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(b_x)

        avg_train_loss = total_loss / len(train_ds)

        # Validation evaluation
        _, _, val_macro_f1, val_bal_acc, val_acc, _ = evaluate_loader(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_macro_f1": val_macro_f1,
            "val_bal_acc": val_bal_acc,
        })

        if stopper.step(val_macro_f1, epoch, model):
            break

    # Restore best checkpoint
    stopper.restore_best_weights(model)

    # Final test evaluation with best weights
    test_preds, test_targets, test_f1, test_bal_acc, test_acc, test_cm = evaluate_loader(
        model, test_loader, device
    )

    budget_info = budget_tracker.end_fold() if budget_tracker else {}

    return {
        "test_macro_f1": float(test_f1),
        "test_balanced_accuracy": float(test_bal_acc),
        "test_accuracy": float(test_acc),
        "test_confusion_matrix": test_cm.tolist(),
        "best_val_macro_f1": float(stopper.best_score),
        "best_epoch": int(stopper.best_epoch),
        "total_epochs": len(history),
        "history": history,
        "test_predictions": test_preds.tolist(),
        "test_targets": test_targets.tolist(),
        "budget": budget_info,
    }
