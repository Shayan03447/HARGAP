"""1D Convolutional Neural Network architecture per Hammerla et al."""

import torch
import torch.nn as nn
from hargap.models.base import HARModel


class CNN1D(HARModel):
    """
    1D-CNN architecture following Hammerla et al. comparison design.
    Features 4 1D conv blocks with batch normalization, ReLU activation,
    adaptive max pooling, dropout 0.5, and a linear classification head.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        conv_channels: list = [64, 64, 128, 128],
        kernel_sizes: list = [5, 5, 3, 3],
        dropout: float = 0.5,
        fc_hidden: int = 128
    ):
        super().__init__(n_channels, n_classes)
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv1d(n_channels, conv_channels[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(inplace=True),
            # Block 3
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2),
            nn.BatchNorm1d(conv_channels[2]),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv1d(conv_channels[2], conv_channels[3], kernel_size=kernel_sizes[3], padding=kernel_sizes[3] // 2),
            nn.BatchNorm1d(conv_channels[3]),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[3], fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes)
        )

        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 128)
        feats = self.encoder(x)  # (B, 128, 128)
        pooled = self.pool(feats).squeeze(-1)  # (B, 128)
        dropped = self.dropout(pooled)
        logits = self.fc(dropped)  # (B, n_classes)
        return logits
