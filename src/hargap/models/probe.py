"""Subject identity probe classifier to quantify biometric leakage in inertial streams."""

import torch
import torch.nn as nn
from hargap.models.base import HARModel


class SubjectIdentityProbe(HARModel):
    """
    Compact subject-identity classifier (3 Conv1D layers + 2 FC layers).
    Trained to predict which subject generated an inertial window,
    quantifying biometric uniqueness and sensor signature leakage.
    """

    def __init__(
        self,
        n_channels: int,
        n_subjects: int,
        conv_channels: list = [32, 64, 128],
        dropout: float = 0.5
    ):
        super().__init__(n_channels, n_subjects)
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, conv_channels[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[2]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(conv_channels[2], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_subjects)
        )

        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 128)
        feats = self.conv(x).squeeze(-1)  # (B, 128)
        logits = self.classifier(feats)  # (B, n_subjects)
        return logits
