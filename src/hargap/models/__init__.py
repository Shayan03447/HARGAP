"""Model architectures and model factory."""

from typing import Dict, Any
from hargap.models.base import HARModel
from hargap.models.cnn import CNN1D
from hargap.models.lstm import LSTMModel
from hargap.models.cnnlstm import DeepConvLSTM
from hargap.models.attention_cnnlstm import AttentionCNN_LSTM
from hargap.models.probe import SubjectIdentityProbe


def build_model(
    name: str,
    n_channels: int,
    n_classes: int,
    **kwargs: Any
) -> HARModel:
    """Factory creating any of the four frozen architectures or the identity probe."""
    name_lower = name.lower()
    if name_lower in ["cnn", "1d_cnn", "1d-cnn"]:
        return CNN1D(n_channels=n_channels, n_classes=n_classes, **kwargs)
    elif name_lower == "lstm":
        return LSTMModel(n_channels=n_channels, n_classes=n_classes, **kwargs)
    elif name_lower in ["cnnlstm", "deepconvlstm", "cnn-lstm"]:
        return DeepConvLSTM(n_channels=n_channels, n_classes=n_classes, **kwargs)
    elif name_lower in ["attn", "attention", "attention_cnnlstm", "attention-cnnlstm"]:
        return AttentionCNN_LSTM(n_channels=n_channels, n_classes=n_classes, **kwargs)
    elif name_lower in ["probe", "identity_probe"]:
        return SubjectIdentityProbe(n_channels=n_channels, n_subjects=n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model architecture: {name}")


__all__ = [
    "HARModel",
    "CNN1D",
    "LSTMModel",
    "DeepConvLSTM",
    "AttentionCNN_LSTM",
    "SubjectIdentityProbe",
    "build_model",
]
