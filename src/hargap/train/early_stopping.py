"""Early stopping mechanism with state preservation based on validation macro-F1."""

from typing import Optional, Dict, Any
import copy
import torch
import torch.nn as nn


class EarlyStopper:
    """
    Early stopping monitor.
    Tracks validation macro-F1 score with a fixed patience of 10 epochs.
    Restores the best checkpoint state when patience is exhausted.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, max_epochs: int = 150):
        self.patience = patience
        self.min_delta = min_delta
        self.max_epochs = max_epochs
        self.best_score = -float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_state_dict: Optional[Dict[str, Any]] = None
        self.stopped_epoch = 0

    def step(self, score: float, epoch: int, model: nn.Module) -> bool:
        """
        Updates the tracker. Returns True if training should stop.
        """
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.patience_counter = 0
            # Clone model state dict
            self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience or epoch >= self.max_epochs:
            self.stopped_epoch = epoch
            return True

        return False

    def restore_best_weights(self, model: nn.Module):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
