# HARGAP: Cross-Subject Generalisation Gap in Wearable HAR

**Deep Learning for Human Activity Recognition from Wearable Inertial Sensors: Quantifying and Reducing the Cross-Subject Generalisation Gap**

COM748 Masters Research Project — Zeeshan Tariq (20107917)  
Supervisor: Adediji Fakoya

This repository implements the end-to-end, leakage-aware experimental pipeline specified in the COM748 code implementation outline. It quantifies the **P1 record-wise vs P2 leave-one-subject-out (LOSO)** generalisation gap, compares baseline deep architectures, and evaluates subject-normalisation/augmentation mitigations (**M1–M3**) under a pre-specified, reproducible protocol.

## Features (aligned to outline/workstreams)

- **Data layer (`hargap.data`)**: deterministic loading/parsing for UCI-HAR, PAMAP2, MHEALTH; resampling (PAMAP2 100→50 Hz), gap healing, segmentation, windowing (128 samples / 2.56 s, 50% overlap), cached window tensors + metadata, synthetic benchmark generator.
- **Protocol layer (`hargap.protocols`)**: P1 stratified record-wise splits (70/15/15, seeds {1,2,3}), P2 LOSO folds with canonical validation panel (`max(1, floor(0.2*(n-1)))` ascending IDs), official UCI 21/9 reference split, hashed fold manifests.
- **Model layer (`hargap.models`)**: 1D-CNN, LSTM, DeepConvLSTM, Attention CNN-LSTM, plus subject-identity probe.
- **Mitigation layer (`hargap.mitigations`)**: baseline global z-score (train-fit), M1 per-subject z-score, M2 sensor augmentation (jitter/scale/3D rotation, train-only), M3 composition (M1 then M2).
- **Training/eval (`hargap.train`, `hargap.eval`)**: shared deterministic training loop, weighted CE (train-only weights), early stopping on validation macro-F1, per-subject metrics, append-only `RESULTS/runs.jsonl`.
- **Statistics/reporting (`hargap.stats`, `hargap.report`)**: Wilcoxon/rank-biserial/bootstrap CI, Friedman/Kendall’s W/Holm, RQ1–RQ3 table/figure generation from result logs.
- **Verification (`hargap.verify`, `tests/`)**: disjointness, window boundaries, fitting/augmentation scope, manifest reproducibility, stats implementation checks, permutation/reproduction/probe design checks.
- **Scripts (`scripts/`)**: data provenance/cache/fold building, experiment execution, analysis, report generation.
- **Demonstrator (`demonstrator/`)**: Streamlit replay, results explorer, integrity panel, and stretch live Phyphox input path.
- **CI (`.github/workflows/ci.yaml`)**: fast verification suite + smoke execution.

## Repository layout

```text
com748-hargap/
  configs/            base.yaml, datasets/*.yaml, models/*.yaml, runs/
  data/               raw/<downloaded>, cache/<window tensors + metadata>
  src/hargap/         config, data, protocols, models, mitigations, train, eval, stats, verify, report
  tests/              verification and design-check tests
  demonstrator/       Streamlit app and panels
  scripts/            CLI entry points for pipeline stages
  MANIFESTS/          fold manifests (*.json) and hashes (*.sha256)
  RESULTS/            runs.jsonl, logs/, checkpoints/
  .github/workflows/  ci.yaml
```

## Environment

- Python `>=3.10`
- Core dependencies: `torch`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `statsmodels`, `pyyaml`, `pyarrow`, `matplotlib`
- Testing: `pytest`
- Demonstrator: `streamlit`
- Optional/stretch: Phyphox websocket client for live input

### Install

```bash
cd com748-hargap
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate

pip install -r requirements-lock.txt
pip install -e .
# or for dev/tests: pip install -e .[dev]
```

`requirements-lock.txt` contains the pinned environment captured for reproducibility.

## Datasets

Raw datasets are **not** redistributed. Download them from the original sources and place under `data/raw/`:

| Dataset | Expected location | Notes | Source |
|---|---|---|---|
| UCI-HAR | `data/raw/ucihar/` | Original `train/`, `test/`, `Inertial Signals/` structure | Anguita et al., ESANN 2013 — https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones |
| PAMAP2 | `data/raw/pamap2/` | `subject101.dat` … `subject109.dat` | Reiss & Stricker, ISWC 2012, doi:10.1109/ISWC.2012.13 — https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring |
| MHEALTH | `data/raw/mhealth/` | `mHealth_subject1.log` … `mHealth_subject10.log` | Banos et al., 2014, doi:10.1007/978-3-319-13105-4_14 — https://archive.ics.uci.edu/dataset/319/mhealth+dataset |

Use the helper to inspect expected paths and record provenance/hashes (manual download may be required by licence terms):

```bash
python scripts/download_data.py --list
python scripts/download_data.py --dataset ucihar --raw-dir data/raw/ucihar --record-hashes
```

## Quick start (synthetic, no external data)

```bash
# from com748-hargap/
python scripts/build_cache.py --dataset synthetic --out data/cache/synthetic_windows.npz
python scripts/build_folds.py --dataset synthetic --cache data/cache/synthetic_windows.npz --out-dir MANIFESTS

# fast verification suite
python -m pytest -q

# single smoke fold/run
python scripts/run_experiment.py --config configs/runs/example_synthetic_cnn_p2_baseline.yaml --smoke

# analysis/report generation (reads RESULTS/runs.jsonl)
python scripts/analyze.py --results RESULTS/runs.jsonl
python scripts/make_report_tables.py --results RESULTS/runs.jsonl --out-dir RESULTS/tables --fig-dir RESULTS/figures
```

## End-to-end pipeline (real datasets)

1. **Record/download raw data and hashes**
   ```bash
   python scripts/download_data.py --dataset ucihar  --raw-dir data/raw/ucihar  --record-hashes
   python scripts/download_data.py --dataset pamap2  --raw-dir data/raw/pamap2  --record-hashes
   python scripts/download_data.py --dataset mhealth --raw-dir data/raw/mhealth --record-hashes
   ```
2. **Build deterministic caches**
   ```bash
   python scripts/build_cache.py --dataset ucihar  --raw-dir data/raw/ucihar  --out data/cache/ucihar_windows.npz
   python scripts/build_cache.py --dataset pamap2  --raw-dir data/raw/pamap2  --out data/cache/pamap2_windows.npz
   python scripts/build_cache.py --dataset mhealth --raw-dir data/raw/mhealth --out data/cache/mhealth_windows.npz
   ```
3. **Build and hash fold manifests** (P1, P2, and UCI official reference)
   ```bash
   python scripts/build_folds.py --dataset ucihar  --cache data/cache/ucihar_windows.npz  --out-dir MANIFESTS
   python scripts/build_folds.py --dataset pamap2  --cache data/cache/pamap2_windows.npz  --out-dir MANIFESTS
   python scripts/build_folds.py --dataset mhealth --cache data/cache/mhealth_windows.npz --out-dir MANIFESTS
   ```
4. **Run verification gate** (must be green before experiment batches)
   ```bash
   python -m pytest -q
   python scripts/run_experiment.py --verify-only --report RESULTS/verify_report.json
   ```
5. **Execute experiment matrix** (resumable; refuses stale/red/missing verification for manifests)
   ```bash
   python scripts/run_experiment.py \
     --matrix core \
     --datasets ucihar pamap2 mhealth \
     --architectures cnn lstm cnnlstm attn \
     --protocols p1 p2 \
     --mitigations baseline m1 m2 m3
   ```
6. **Analyse results and generate tables/figures**
   ```bash
   python scripts/analyze.py --results RESULTS/runs.jsonl --budget-hours 40
   python scripts/make_report_tables.py --results RESULTS/runs.jsonl --out-dir RESULTS/tables --fig-dir RESULTS/figures
   ```

## Reproduce one fold end-to-end

```bash
python scripts/build_cache.py --dataset synthetic --out data/cache/synthetic_windows.npz
python scripts/build_folds.py --dataset synthetic --cache data/cache/synthetic_windows.npz --out-dir MANIFESTS
python scripts/run_experiment.py --config configs/runs/example_synthetic_cnn_p2_baseline.yaml
```

Run records in `RESULTS/runs.jsonl` include config/manifest hashes, git commit (when available), seeds, metrics, per-subject scores, and timing/memory budget.

## Experimental specification highlights

- **Windows:** 128 samples @ 50 Hz, stride 64 (50% overlap); UCI uses supplied windows.
- **Protocols:** P1 pooled stratified 70/15/15 (seeds 1/2/3); P2 LOSO with deterministic 20% ascending-ID validation panel; UCI official 21/9 read-only reference.
- **Architectures:** 1D-CNN, LSTM, DeepConvLSTM, Attention CNN-LSTM (published-derived configs recorded in YAML).
- **Mitigations:** Baseline global z-score (train-fit), M1 per-subject z-score, M2 train-only augmentation (jitter σ=0.05, scale N(1,0.1), rotation ±30° per sensor), M3 = M1→M2.
- **Training:** Adam lr 1e-3, weighted CE (train-only inverse-frequency, mean-normalised), batch 64, early stopping on val macro-F1 (patience 10, cap 150), deterministic seeding.
- **Statistics:** RQ1 one-sided Wilcoxon + rank-biserial + bootstrap 95% CI (10k, seeded); RQ2 Friedman/Kendall’s W + Holm post-hoc; RQ3 triple criterion with Holm families.

## Demonstrator

```bash
streamlit run demonstrator/app.py
```

- **Replay (DR-1):** held-out cached windows with P1 vs LOSO checkpoint predictions and running macro-F1 (when checkpoints/manifests available).
- **Results explorer (DR-2):** per-subject distributions, confusion heatmaps, mitigation effect summaries from `RESULTS/runs.jsonl`.
- **Integrity panel (DR-3):** verification suite execution/output and manifest hash display.
- **Live phone stream (DR-4, stretch):** Phyphox websocket sliding-window inference (config-gated).

## Testing and CI

```bash
# fast verification/design suite
python -m pytest -q
```

CI: `.github/workflows/ci.yaml` installs pinned deps and runs the fast suite plus a smoke path on push/PR.

## Reproducibility and integrity notes

- `run_experiment.py` refuses batches when verification is missing/stale/red relative to manifests (`must_be_green` gate).
- Manifests, configs, cache hashes, seeds, and commit IDs are recorded per run.
- `data/raw`, `data/cache`, manifests, logs, and checkpoints are git-ignored by default; publish hashed artefacts/results as required for the dissertation record.
- Stretch options (extra datasets/seeds, 60 s M1 calibration, live stream) are off by default.

## License

MIT (see `LICENSE`). Dataset licences/terms are governed by their original providers.
