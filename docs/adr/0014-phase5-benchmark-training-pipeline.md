# ADR-0014: Phase 5 — Benchmark Training Pipeline

**Status:** Accepted  
**Date:** 2026-08-08

## Context

MVPDR+ has been developed with projected accuracy numbers. To validate the system and produce real benchmark results for the portfolio, we need a reproducible training pipeline that runs the full experiment matrix across datasets, shot settings, backbones, and component ablations.

The development machine lacks a GPU, so the pipeline must be designed to run on cloud GPU environments (Google Colab, Kaggle).

## Decisions

### D1: Dataset Acquisition Strategy

- **PlantDoc** (27 classes): Auto-downloaded from GitHub via git clone
- **PlantVillage** (38 classes): Downloaded via Kaggle CLI or manual download
- Expected directory structure: `<root>/<dataset>/images/<class_name>/<images>`
- Download script handles deduplication and idempotent re-runs

### D2: Experiment Matrix

The complete experiment suite covers:

| Stage | Runs | Description |
|-------|------|-------------|
| Zero-shot | 4 backbones × 2 datasets = 8 | CLIP baseline without training |
| MVPDR baseline | 4 shots × 2 datasets × 1 seed = 8 | Original MVPDR model |
| MVPDR+ | 4 shots × 2 datasets × 1 seed = 8 | Full model with all components |
| Ablation | 7 configs × 1 seed = 7 | Component contribution analysis |
| Backbone | 4 backbones × 1 seed = 4 | Architecture comparison |

Total: ~35 runs (single seed), ~105 with 3 seeds for confidence intervals.

### D3: Orchestration Design

- `scripts/run_experiments.py` orchestrates all stages with a unified CLI
- Temporary configs generated with overrides (root_path, shots, component toggles)
- Each stage can be run independently via `--stage`
- Supports `--dry_run` for validation before committing GPU time
- Results saved to `results/<dataset>/<backbone>/<variant>/results.json`

### D4: Results Aggregation

- `scripts/aggregate_results.py` scans all results.json files
- Produces markdown summary tables, comparison bar charts, few-shot learning curves
- Designed to be re-run as new experiments complete

### D5: Colab Notebook

- `notebooks/train_mvpdr.ipynb` provides a ready-to-run GPU training environment
- Structured as sequential cells: setup → download → zero-shot → baseline → MVPDR+ → ablation → aggregate
- Results downloadable as tarball
- Estimated runtime: ~2-4 hours on T4 for full suite with 3 seeds

## Artifacts Created

- `scripts/download_datasets.py` — dataset acquisition
- `scripts/evaluate_zeroshot.py` — zero-shot CLIP evaluation
- `scripts/run_experiments.py` — experiment orchestration
- `scripts/aggregate_results.py` — results collection and visualization
- `notebooks/train_mvpdr.ipynb` — Colab training notebook
