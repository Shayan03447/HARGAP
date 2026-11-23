"""DeepConvLSTM hybrid architecture per Ordóñez and Roggen."""

import torch
import torch.nn as nn
from hargap.models.base import HARModel


class DeepConvLSTM(HARModel):
    """
    Hybrid Convolutional-Recurrent architecture (Ordóñez and Roggen).
    Consists of 4 1D convolutional layers extracting local temporal feature maps,
    followed by LSTM layers for recurrent sequential modeling, dropout, and head.
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
        dropout: float = 0.5,
        bidirectional: bool = False,
        **kwargs,
    ):
        super().__init__(n_channels, n_classes)

        # Normalize conv params (support yaml list forms and int forms)
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

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, c_out, kernel_size=k, padding=padding),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(c_out, c_out, kernel_size=k, padding=padding),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(c_out, c_out, kernel_size=k, padding=padding),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
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

        self.dropout = nn.Dropout(float(dropout))
        self.fc = nn.Linear(int(lstm_hidden) * self.num_directions, n_classes)

        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        c = self.conv1(x)
        c = self.conv2(c)
        c = self.conv3(c)
        c = self.conv4(c)  # (B, conv_channels, T)

        # (B, T, conv_channels)
        seq = c.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(seq)

        if self.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]

        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)
        return logits
