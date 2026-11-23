"""Recurrent LSTM architecture per Hammerla et al."""

import torch
import torch.nn as nn
from hargap.models.base import HARModel


class LSTMModel(HARModel):
    """
    Recurrent LSTM architecture following Hammerla et al.
    Consumes (B, C, 128) reshaped to (B, 128, C).
    2-layer Bidirectional LSTM, dropout 0.5, and linear classification head.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.5,
        fc_hidden: int = 128
    ):
        super().__init__(n_channels, n_classes)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        encoder_dim = hidden_size * self.num_directions
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(encoder_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes)
        )

        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 128) -> (B, 128, C)
        x_seq = x.permute(0, 2, 1)
        out, (h_n, c_n) = self.lstm(x_seq)

        # Pool over sequence or use last hidden state
        if self.bidirectional:
            last_feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_feat = h_n[-1]

        dropped = self.dropout(last_feat)
        logits = self.fc(dropped)
        return logits
