"""Configuration management and validation for experiment execution."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import hashlib
import json
import yaml
from pathlib import Path


@dataclass
class WindowConfig:
    length: int = 128
    overlap: float = 0.5
    stride: int = 64
    source: str = "segmented_windows"


@dataclass
class ProtocolConfig:
    kind: str = "p2"  # "p1", "p2", "official"
    fold_id: int = 0
    seed: int = 1
    manifest_sha256: str = ""
    validation_rule: str = "fixed 20% panel, min 1, ascending subject id"


@dataclass
class ModelConfig:
    name: str = "cnn"
    config_path: str = "models/cnn.yaml"
    n_channels: int = 9
    n_classes: int = 6
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    optimizer: str = "adam"
    lr: float = 1e-3
    adam_betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    adam_eps: float = 1e-8
    batch_size: int = 64
    epochs_max: int = 150
    patience: int = 10
    loss: str = "weighted_ce"
    weights_from: str = "train_only"
    amp: bool = False
    num_workers: int = 0


@dataclass
class MitigationConfig:
    name: str = "baseline"  # "baseline", "m1", "m2", "m3"
    transforms: List[str] = field(default_factory=list)
    p_each: float = 0.5
    jitter_sigma: float = 0.05
    scale_mean: float = 1.0
    scale_std: float = 0.1
    rotation_deg_max: float = 30.0
    applies_to: str = "train_only"


@dataclass
class SeedConfig:
    python: int = 1
    numpy: int = 11
    torch: int = 111


@dataclass
class VerifyConfig:
    report: str = ""
    must_be_green: bool = True


@dataclass
class RunConfig:
    run_id: str
    dataset: str
    channels: List[str]
    window: WindowConfig
    protocol: ProtocolConfig
    model: ModelConfig
    train: TrainConfig
    mitigation: MitigationConfig
    seeds: SeedConfig
    verify: VerifyConfig
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "channels": self.channels,
            "window": {
                "length": self.window.length,
                "overlap": self.window.overlap,
                "stride": self.window.stride,
                "source": self.window.source,
            },
            "protocol": {
                "kind": self.protocol.kind,
                "fold_id": self.protocol.fold_id,
                "seed": self.protocol.seed,
                "manifest_sha256": self.protocol.manifest_sha256,
                "validation_rule": self.protocol.validation_rule,
            },
            "model": {
                "name": self.model.name,
                "config_path": self.model.config_path,
                "n_channels": self.model.n_channels,
                "n_classes": self.model.n_classes,
                "params": self.model.params,
            },
            "train": {
                "optimizer": self.train.optimizer,
                "lr": self.train.lr,
                "adam_betas": self.train.adam_betas,
                "adam_eps": self.train.adam_eps,
                "batch_size": self.train.batch_size,
                "epochs_max": self.train.epochs_max,
                "patience": self.train.patience,
                "loss": self.train.loss,
                "weights_from": self.train.weights_from,
                "amp": self.train.amp,
                "num_workers": self.train.num_workers,
            },
            "mitigation": {
                "name": self.mitigation.name,
                "transforms": self.mitigation.transforms,
                "p_each": self.mitigation.p_each,
                "jitter_sigma": self.mitigation.jitter_sigma,
                "scale_mean": self.mitigation.scale_mean,
                "scale_std": self.mitigation.scale_std,
                "rotation_deg_max": self.mitigation.rotation_deg_max,
                "applies_to": self.mitigation.applies_to,
            },
            "seeds": {
                "python": self.seeds.python,
                "numpy": self.seeds.numpy,
                "torch": self.seeds.torch,
            },
            "verify": {
                "report": self.verify.report,
                "must_be_green": self.verify.must_be_green,
            },
            "extra": self.extra,
        }

    def compute_sha256(self) -> str:
        serialized = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def run_config_from_dict(d: Dict[str, Any]) -> RunConfig:
    """Constructs RunConfig from a (possibly nested/partial) mapping."""
    w = _get(d, "window", {}) or {}
    p = _get(d, "protocol", {}) or {}
    m = _get(d, "model", {}) or {}
    t = _get(d, "train", {}) or {}
    mig = _get(d, "mitigation", {}) or {}
    s = _get(d, "seeds", {}) or {}
    v = _get(d, "verify", {}) or {}

    window = WindowConfig(
        length=int(_get(w, "length", 128)),
        overlap=float(_get(w, "overlap", 0.5)),
        stride=int(_get(w, "stride", 64)),
        source=str(_get(w, "source", "segmented_windows")),
    )
    protocol = ProtocolConfig(
        kind=str(_get(p, "kind", "p2")),
        fold_id=int(_get(p, "fold_id", 0)),
        seed=int(_get(p, "seed", _get(s, "protocol", 1))),
        manifest_sha256=str(_get(p, "manifest_sha256", "")),
        validation_rule=str(_get(p, "validation_rule", "fixed 20% panel, min 1, ascending subject id")),
    )
    model = ModelConfig(
        name=str(_get(m, "name", "cnn")),
        config_path=str(_get(m, "config_path", _get(m, "config", f"configs/models/{_get(m,'name','cnn')}.yaml"))),
        n_channels=int(_get(m, "n_channels", _get(d, "n_channels", 9))),
        n_classes=int(_get(m, "n_classes", _get(d, "n_classes", 6))),
        params=dict(_get(m, "params", {})),
    )
    train = TrainConfig(
        optimizer=str(_get(t, "optimizer", _get(t, "opt", "adam"))),
        lr=float(_get(t, "lr", _get(t, "learning_rate", 1e-3))),
        adam_betas=list(_get(t, "adam_betas", [0.9, 0.999])),
        adam_eps=float(_get(t, "adam_eps", 1e-8)),
        batch_size=int(_get(t, "batch_size", _get(t, "batch", 64))),
        epochs_max=int(_get(t, "epochs_max", _get(t, "epochs", 150))),
        patience=int(_get(t, "patience", 10)),
        loss=str(_get(t, "loss", "weighted_ce")),
        weights_from=str(_get(t, "weights_from", "train_only")),
        amp=bool(_get(t, "amp", False)),
        num_workers=int(_get(t, "num_workers", 0)),
    )
    mitigation = MitigationConfig(
        name=str(_get(mig, "name", "baseline")),
        transforms=list(_get(mig, "transforms", [])),
        p_each=float(_get(mig, "p_each", 0.5)),
        jitter_sigma=float(_get(mig, "jitter_sigma", 0.05)),
        scale_mean=float(_get(mig, "scale_mean", 1.0)),
        scale_std=float(_get(mig, "scale_std", 0.1)),
        rotation_deg_max=float(_get(mig, "rotation_deg_max", 30.0)),
        applies_to=str(_get(mig, "applies_to", "train_only")),
    )
    seeds = SeedConfig(
        python=int(_get(s, "python", 1)),
        numpy=int(_get(s, "numpy", 11)),
        torch=int(_get(s, "torch", 111)),
    )
    verify = VerifyConfig(
        report=str(_get(v, "report", "")),
        must_be_green=bool(_get(v, "must_be_green", True)),
    )
    extra = {k: val for k, val in d.items() if k not in {"run_id", "dataset", "channels", "window", "protocol", "model", "train", "mitigation", "seeds", "verify", "n_channels", "n_classes"}}
    run_id = str(_get(d, "run_id", "run"))
    dataset = str(_get(d, "dataset", "synthetic"))
    channels = list(_get(d, "channels", []))
    return RunConfig(
        run_id=run_id,
        dataset=dataset,
        channels=channels,
        window=window,
        protocol=protocol,
        model=model,
        train=train,
        mitigation=mitigation,
        seeds=seeds,
        verify=verify,
        extra=extra,
    )


def load_run_config(path: str | Path) -> RunConfig:
    data = load_yaml(path)
    cfg = run_config_from_dict(data)
    validate_config(cfg)
    return cfg


def validate_config(cfg: RunConfig) -> bool:
    assert cfg.dataset in ["ucihar", "pamap2", "mhealth", "synthetic"], f"Invalid dataset {cfg.dataset}"
    assert cfg.protocol.kind in ["p1", "p2", "official"], f"Invalid protocol {cfg.protocol.kind}"
    assert cfg.model.name in ["cnn", "lstm", "cnnlstm", "attn", "probe"], f"Invalid model {cfg.model.name}"
    assert cfg.mitigation.name in ["baseline", "m1", "m2", "m3"], f"Invalid mitigation {cfg.mitigation.name}"
    assert cfg.window.length == 128, "Window length must be 128 samples (2.56s at 50Hz)"
    assert cfg.train.batch_size > 0, "Batch size must be positive"
    return True
