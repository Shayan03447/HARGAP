"""Base model abstraction for wearable HAR architectures."""

import abc
import torch
import torch.nn as nn
from typing import Dict, Any


class HARModel(nn.Module, abc.ABC):
    """
    Unified abstract interface for all deep learning models.
    Input tensor shape: (Batch, Channels, 128)
    Output tensor shape: (Batch, n_classes)
    """

    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits."""
        pass

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def initialize_weights(self):
        """Kaiming (He) normal initialization for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
