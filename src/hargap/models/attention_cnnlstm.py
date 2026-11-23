"""Attention-augmented DeepConvLSTM architecture per Murahari and Plötz."""

import torch
import torch.nn as nn
from hargap.models.base import HARModel


class TemporalAttention(nn.Module):
    """
    Computes scalar alignment weights over recurrent sequence states.
    Murahari and Plötz formulation: alpha_t = softmax(v^T tanh(W h_t + b)).
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(self, lstm_outputs: torch.Tensor) -> torch.Tensor:
        # lstm_outputs: (B, T, hidden_dim)
        scores = self.projection(lstm_outputs)  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        context = torch.sum(lstm_outputs * weights, dim=1)  # (B, hidden_dim)
        return context


class AttentionCNN_LSTM(HARModel):
    """
    Attention-augmented DeepConvLSTM architecture (Murahari and Plötz).
    Combines 4-layer 1D CNN feature extraction, LSTM temporal integration,
    and temporal self-attention over recurrent states.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        conv_channels=64,
        conv_kernel=5,
        kernel_sizes=None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        num_layers: int = None,
        attention_dim: int = 64,
        dropout: float = 0.5,
        bidirectional: bool = False,
        **kwargs,
    ):
        super().__init__(n_channels, n_classes)

        # Normalize conv params
        if isinstance(conv_channels, (list, tuple)):
            c_out = int(conv_channels[0]) if conv_channels else 64
        else:
            c_out = int(conv_channels)
        ks = kernel_sizes if kernel_sizes is not None else conv_kernel
        if isinstance(ks, (list, tuple)):
            k = int(ks[0]) if ks else 5
        else:
            k = int(ks)
        if num_layers is not None and lstm_layers == 2:
            lstm_layers = int(num_layers)

        padding = k // 2

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, c_out, kernel_size=k, padding=padding),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, kernel_size=k, padding=padding),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, kernel_size=k, padding=padding),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, kernel_size=k, padding=padding),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )

        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=c_out,
            hidden_size=int(lstm_hidden),
            num_layers=int(lstm_layers),
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
        )

        hidden_dim = int(lstm_hidden) * self.num_directions
        self.attention = TemporalAttention(hidden_dim, int(attention_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.fc = nn.Linear(hidden_dim, n_classes)

        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        feats = self.conv(x)  # (B, conv_channels, T)
        seq = feats.permute(0, 2, 1)  # (B, T, conv_channels)

        lstm_out, _ = self.lstm(seq)  # (B, T, hidden_dim)
        context = self.attention(lstm_out)  # (B, hidden_dim)

        dropped = self.dropout(context)
        logits = self.fc(dropped)
        return logits
