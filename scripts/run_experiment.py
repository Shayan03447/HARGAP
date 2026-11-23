"""Execute a single experiment run or matrix from manifests and cached data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hargap.config import RunConfig, load_run_config, load_yaml, run_config_from_dict, validate_config
from hargap.data.cache import WindowStore
from hargap.eval.database import ResultDB
from hargap.eval.metrics import aggregate_per_subject_scores
from hargap.mitigations.augmentor import SensorAugmentor
from hargap.mitigations.compose import MitigationPipeline
from hargap.mitigations.normalizer import GlobalNormalizer, SubjectNormalizer
from hargap.models import build_model
from hargap.protocols.manifest import FoldManifest
from hargap.train.budget import BudgetTracker
from hargap.train.trainer import train_fold
from hargap.verify.suite import run_verification_suite


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return "nogit"


def _ensure_verification(cfg: RunConfig, manifest: Optional[FoldManifest], allow_unverified: bool, report_path: Optional[str]) -> Dict[str, Any]:
    if allow_unverified or not cfg.verify.must_be_green:
        return {"status": "SKIPPED", "is_green": False, "reason": "allow_unverified"}
    rp = Path(report_path or cfg.verify.report or "RESULTS/verify_report.json")
    if not rp.exists():
        raise RuntimeError(f"Verification report missing at {rp}. Run with --verify-only or provide green report.")
    with open(rp, "r", encoding="utf-8") as f:
        report = json.load(f)
    if not report.get("is_green") and report.get("status") != "GREEN":
        raise RuntimeError(f"Verification report at {rp} is not GREEN.")
    # optional staleness marker
    rep_manifest = report.get("manifest_sha256")
    if manifest and rep_manifest and rep_manifest not in ("", manifest.compute_sha256()):
        # allow aggregate reports; strict refusal only when explicitly mismatched single-manifest report
        if report.get("single_manifest"):
            raise RuntimeError(f"Verification report manifest hash {rep_manifest} does not match {manifest.compute_sha256()}")
    return report


def _run_verify_only(args) -> int:
    report_path = args.report or "RESULTS/verify_report.json"
    report = run_verification_suite(report_output_path=report_path)
    print(json.dumps(report, indent=2))
    return 0 if report.get("is_green") else 1


def _default_cache_for(dataset: str) -> Path:
    return ROOT / "data" / "cache" / f"{dataset}_windows.npz"


def _find_manifest(cfg: RunConfig, manifests_dir: Path) -> Path:
    p = cfg.extra.get("manifest_path") if isinstance(cfg.extra, dict) else None
    if p:
        mp = Path(p)
        if not mp.is_absolute():
            mp = ROOT / mp
        if mp.exists():
            return mp
    d = cfg.dataset
    kind = cfg.protocol.kind
    fold = cfg.protocol.fold_id
    seed = cfg.protocol.seed
    candidates = []
    if kind == "p1":
        candidates.append(manifests_dir / f"{d}_p1_seed{seed}.json")
    if kind == "official":
        candidates.append(manifests_dir / f"{d}_official_fold00_seed{seed}.json")
    candidates.append(manifests_dir / f"{d}_{kind}_fold{fold:02d}_seed{seed}.json")
    candidates.append(manifests_dir / f"{d}_{kind}_fold{fold}_seed{seed}.json")
    for c in candidates:
        if c.exists():
            return c
    hits = sorted(manifests_dir.glob(f"{d}_{kind}*fold*{fold:02d}*.json")) or sorted(manifests_dir.glob(f"{d}_{kind}*fold*{fold}*.json"))
    if hits:
        return hits[0]
    if kind == "p1":
        hits = sorted(manifests_dir.glob(f"{d}_p1*seed*{seed}*.json"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No manifest found for {d}/{kind}/fold{fold}/seed{seed} in {manifests_dir}")


def _build_mitigation(name: str, train_x, train_y, train_subs, val_x, val_subs, test_x, test_subs, seed: int):
    name = name.lower()
    global_norm = GlobalNormalizer()
    subject_norm = SubjectNormalizer()
    augmentor = SensorAugmentor(p_each=0.5, jitter_sigma=0.05, scale_mean=1.0, scale_std=0.1, rotation_deg_max=30.0, random_state=seed)

    global_norm.fit(train_x)
    if name in ("m1", "m3"):
        subject_norm.fit_subjects_batch(train_x, train_subs)
        if len(val_subs):
            subject_norm.fit_subjects_batch(val_x, val_subs)
        if len(test_subs):
            subject_norm.fit_subjects_batch(test_x, test_subs)

    pipeline = MitigationPipeline(name=name, global_normalizer=global_norm, subject_normalizer=subject_norm, augmentor=augmentor)
    augmentor.set_seed(seed)
    proc_train_x = pipeline.process_train(train_x, train_subs)
    proc_val_x = pipeline.process_eval(val_x, val_subs) if len(val_x) else val_x
    proc_test_x = pipeline.process_eval(test_x, test_subs) if len(test_x) else test_x
    return proc_train_x, proc_val_x, proc_test_x


def run_single(cfg: RunConfig, args) -> Dict[str, Any]:
    manifests_dir = Path(args.manifests_dir or (ROOT / "MANIFESTS"))
    if not manifests_dir.is_absolute():
        manifests_dir = ROOT / manifests_dir
    manifest_path = _find_manifest(cfg, manifests_dir)
    manifest = FoldManifest.load(manifest_path)

    cache_path = Path(args.cache) if getattr(args, "cache", None) else _default_cache_for(cfg.dataset)
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    store = WindowStore.load(cache_path)

    _ensure_verification(cfg, manifest, allow_unverified=bool(getattr(args, "allow_unverified", False)), report_path=getattr(args, "report", None))

    def subset(idxs):
        idxs = np.array(idxs, dtype=np.int64)
        if len(idxs) == 0:
            return (np.zeros((0, store.n_channels, store.window_length), dtype=np.float32),
                    np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        return store.get_subset(idxs)

    train_x, train_y, train_subs = subset(manifest.train_indices)
    val_x, val_y, val_subs = subset(manifest.val_indices)
    test_x, test_y, test_subs = subset(manifest.test_indices)

    n_channels = cfg.model.n_channels or store.n_channels
    n_classes = cfg.model.n_classes or int(len(np.unique(store.metadata["activity_code"].values)))

    proc_train_x, proc_val_x, proc_test_x = _build_mitigation(
        cfg.mitigation.name, train_x, train_y, train_subs, val_x, val_subs, test_x, test_subs, seed=cfg.seeds.torch
    )

    model_params: Dict[str, Any] = {}
    model_cfg_path = Path(cfg.model.config_path)
    if not model_cfg_path.is_absolute():
        model_cfg_path = ROOT / model_cfg_path
    if model_cfg_path.exists():
        my = load_yaml(model_cfg_path) or {}
        for k in ("conv_channels", "kernel_sizes", "dropout", "fc_hidden", "hidden_size", "num_layers", "bidirectional", "lstm_hidden", "lstm_layers", "attention_dim", "conv_kernel"):
            if k in my:
                model_params[k] = my[k]
        if "conv_layers" in my and isinstance(my["conv_layers"], list) and my["conv_layers"]:
            cls = my["conv_layers"]
            model_params.setdefault("conv_channels", [c.get("out_channels", 64) for c in cls])
            model_params.setdefault("kernel_sizes", [c.get("kernel_size", 5) for c in cls])
    model_params.update(cfg.model.params or {})

    model = build_model(cfg.model.name, n_channels=n_channels, n_classes=n_classes, **model_params)
    model.initialize_weights()

    device = torch.device("cuda" if torch.cuda.is_available() and not getattr(args, "cpu", False) else "cpu")
    budget = BudgetTracker()

    epochs_max = 1 if getattr(args, "smoke", False) else cfg.train.epochs_max
    patience = 1 if getattr(args, "smoke", False) else cfg.train.patience

    if len(val_x) == 0:
        val_x, val_y = proc_train_x[: max(1, len(proc_train_x)//10)], train_y[: max(1, len(train_y)//10)]
    if len(test_x) == 0:
        test_x, test_y = val_x, val_y

    result = train_fold(
        model=model,
        train_x=proc_train_x,
        train_y=train_y,
        val_x=proc_val_x if len(proc_val_x) else val_x,
        val_y=val_y,
        test_x=proc_test_x if len(proc_test_x) else test_x,
        test_y=test_y,
        n_classes=n_classes,
        batch_size=cfg.train.batch_size,
        lr=cfg.train.lr,
        epochs_max=epochs_max,
        patience=patience,
        seeds={"python": cfg.seeds.python, "numpy": cfg.seeds.numpy, "torch": cfg.seeds.torch},
        use_amp=bool(cfg.train.amp),
        device=device,
        budget_tracker=budget,
    )

    test_preds = np.array(result.get("test_predictions", []), dtype=np.int64)
    test_targets = np.array(result.get("test_targets", []), dtype=np.int64)
    per_subject: Dict[str, float] = {}
    if len(test_preds) == len(test_targets) == len(test_subs) and len(test_subs) > 0:
        per_subject = {str(int(k)): float(v) for k, v in aggregate_per_subject_scores(test_targets, test_preds, test_subs).items()}
    elif len(manifest.test_subjects) == 1:
        per_subject = {str(int(manifest.test_subjects[0])): float(result.get("test_macro_f1", 0.0))}

    ckpt_dir = Path(getattr(args, "checkpoints_dir", None) or (ROOT / "RESULTS" / "checkpoints"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg.run_id}.pt"
    try:
        torch.save({"model_state_dict": model.state_dict(), "config": cfg.to_dict(), "manifest_sha256": manifest.compute_sha256()}, ckpt_path)
    except Exception:
        ckpt_path = None  # type: ignore

    run_record = {
        "run_id": cfg.run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "dataset": cfg.dataset,
        "architecture": cfg.model.name,
        "protocol": cfg.protocol.kind,
        "fold_id": int(manifest.fold_id),
        "seed": int(cfg.protocol.seed),
        "mitigation": cfg.mitigation.name,
        "config_sha256": cfg.compute_sha256(),
        "manifest_sha256": manifest.compute_sha256(),
        "cache_sha256": store.sha256,
        "manifest_path": str(manifest_path),
        "cache_path": str(cache_path),
        "checkpoint_path": str(ckpt_path) if ckpt_path else None,
        "metrics": {
            "macro_f1": float(result.get("test_macro_f1", 0.0)),
            "balanced_accuracy": float(result.get("test_balanced_accuracy", 0.0)),
            "accuracy": float(result.get("test_accuracy", 0.0)),
            "confusion_matrix": result.get("test_confusion_matrix", []),
        },
        "per_subject_macro_f1": per_subject,
        "best_epoch": int(result.get("best_epoch", 0)),
        "total_epochs": int(result.get("total_epochs", 0)),
        "best_val_macro_f1": float(result.get("best_val_macro_f1", 0.0)),
        "budget": result.get("budget", {}),
        "device": str(device),
        "smoke": bool(getattr(args, "smoke", False)),
    }

    db_path = Path(getattr(args, "results", None) or (ROOT / "RESULTS" / "runs.jsonl"))
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    db = ResultDB(db_path)
    db.append(run_record)
    print(json.dumps({"status": "ok", "run_id": cfg.run_id, "macro_f1": run_record["metrics"]["macro_f1"], "results": str(db_path)}))
    return run_record


def _load_completed_run_ids(results_path: Path) -> set:
    if not results_path.exists():
        return set()
    ids = set()
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rid = r.get("run_id")
                if rid:
                    ids.add(str(rid))
            except Exception:
                pass
    return ids


def _dataset_dims(dataset: str) -> Tuple[int, int]:
    cfg_path = ROOT / "configs" / "datasets" / f"{dataset}.yaml"
    if cfg_path.exists():
        d = load_yaml(cfg_path) or {}
        return int(d.get("n_channels", 9)), int(d.get("n_classes", 6))
    defaults = {"synthetic": (9, 6), "ucihar": (9, 6), "pamap2": (18, 12), "mhealth": (15, 12)}
    return defaults.get(dataset, (9, 6))


def _iter_matrix_runs(args) -> Iterable[RunConfig]:
    manifests_dir = Path(args.manifests_dir or (ROOT / "MANIFESTS"))
    if not manifests_dir.is_absolute():
        manifests_dir = ROOT / manifests_dir
    datasets = args.datasets or ["ucihar", "pamap2", "mhealth"]
    architectures = args.architectures or ["cnn", "lstm", "cnnlstm", "attn"]
    protocols = args.protocols or ["p1", "p2"]
    mitigations = args.mitigations or ["baseline", "m1", "m2", "m3"]

    for dataset in datasets:
        n_channels, n_classes = _dataset_dims(dataset)
        cache_path = Path(args.cache) if getattr(args, "cache", None) else _default_cache_for(dataset)
        if not cache_path.is_absolute():
            cache_path = ROOT / cache_path
        # if cache missing, still can enumerate from manifests? we require cache at run time; skip dataset if explicitly missing and not dry-run maybe
        for protocol in protocols:
            pattern = f"{dataset}_{protocol}*.json"
            manifests = sorted(manifests_dir.glob(pattern))
            if not manifests:
                continue
            for mpath in manifests:
                try:
                    md = json.loads(mpath.read_text(encoding="utf-8"))
                except Exception:
                    continue
                fold_id = int(md.get("fold_id", 0))
                seed = int(md.get("seed", 1))
                for arch in architectures:
                    for mit in mitigations:
                        run_id = f"{dataset}__{arch}__{protocol}__{mit}__fold{fold_id:02d}__seed{seed}"
                        cfg = run_config_from_dict({
                            "run_id": run_id,
                            "dataset": dataset,
                            "n_channels": n_channels,
                            "n_classes": n_classes,
                            "protocol": {"kind": protocol, "fold_id": fold_id, "seed": seed, "manifest_path": str(mpath.relative_to(ROOT)) if str(mpath).startswith(str(ROOT)) else str(mpath)},
                            "model": {"name": arch, "config_path": f"configs/models/{arch}.yaml", "n_channels": n_channels, "n_classes": n_classes},
                            "train": {"optimizer": "adam", "lr": 1e-3, "batch_size": 64, "epochs_max": 150, "patience": 10, "loss": "weighted_ce", "weights_from": "train_only", "amp": False},
                            "mitigation": {"name": mit},
                            "seeds": {"python": 1, "numpy": 11, "torch": 111},
                            "verify": {"report": getattr(args, "report", None) or "RESULTS/verify_report.json", "must_be_green": not bool(getattr(args, "allow_unverified", False))},
                            "window": {"length": 128, "overlap": 0.5, "stride": 64},
                        })
                        validate_config(cfg)
                        yield cfg


def run_matrix(args) -> int:
    results_path = Path(args.results or (ROOT / "RESULTS" / "runs.jsonl"))
    if not results_path.is_absolute():
        results_path = ROOT / results_path
    completed = _load_completed_run_ids(results_path)
    state_path = Path(getattr(args, "state_file", None) or (ROOT / "RESULTS" / "logs" / "run_state.json"))
    if not state_path.is_absolute():
        state_path = ROOT / state_path
    state_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    done = 0
    skipped = 0
    failed = 0
    max_runs = int(getattr(args, "max_runs", 0) or 0)

    for cfg in _iter_matrix_runs(args):
        total += 1
        if cfg.run_id in completed:
            skipped += 1
            continue
        if getattr(args, "dry_run", False):
            print(json.dumps({"status": "dry_run", "run_id": cfg.run_id}))
            done += 1
        else:
            try:
                run_single(cfg, args)
                completed.add(cfg.run_id)
                done += 1
            except Exception as e:
                failed += 1
                print(json.dumps({"status": "error", "run_id": cfg.run_id, "error": str(e)}), file=sys.stderr)
                if not getattr(args, "keep_going", False):
                    break
        if max_runs and (done + failed) >= max_runs:
            break

        # persist state
        state = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "completed_count": len(completed), "done_this_invocation": done, "skipped": skipped, "failed": failed, "results_path": str(results_path)}
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    summary = {"status": "ok" if failed == 0 else "partial", "enumerated": total, "executed": done, "skipped_already_complete": skipped, "failed": failed, "state_file": str(state_path)}
    print(json.dumps(summary, indent=2))
    return 0 if failed == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Path to run config YAML")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--report", help="Verification report path")
    ap.add_argument("--allow-unverified", action="store_true", help="Bypass green verification gate (smoke/dev only)")
    ap.add_argument("--smoke", action="store_true", help="Run 1 epoch for smoke testing")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cache", help="Cache .npz base path override (single dataset/run)")
    ap.add_argument("--manifests-dir", default="MANIFESTS")
    ap.add_argument("--results", default="RESULTS/runs.jsonl")
    ap.add_argument("--checkpoints-dir", default="RESULTS/checkpoints")
    ap.add_argument("--matrix", choices=["core"], help="Run core matrix from manifests")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--architectures", nargs="*")
    ap.add_argument("--protocols", nargs="*")
    ap.add_argument("--mitigations", nargs="*")
    ap.add_argument("--state-file", help="Resumable matrix run-state JSON path")
    ap.add_argument("--max-runs", type=int, default=0, help="Limit runs executed this invocation (0=all pending)")
    ap.add_argument("--dry-run", action="store_true", help="Enumerate matrix runs without executing")
    ap.add_argument("--keep-going", action="store_true", help="Continue matrix after a failed run")
    args = ap.parse_args()

    if args.verify_only:
        return _run_verify_only(args)

    if args.matrix:
        # ensure verify gate can pass if required: user should run --verify-only first or pass --allow-unverified
        return run_matrix(args)

    if not args.config:
        ap.error("--config is required unless --verify-only or --matrix is set")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_run_config(cfg_path)
    raw = load_yaml(cfg_path) or {}
    if "protocol" in raw and isinstance(raw["protocol"], dict) and "manifest_path" in raw["protocol"]:
        cfg.extra["manifest_path"] = raw["protocol"]["manifest_path"]
    run_single(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())