# ADR-0001: Package Restructuring from src/ to mvpdr/

**Status:** Accepted  
**Date:** 2026-08-07

## Context

The original codebase lived in `src/` with flat imports (`import clip`, `from datasets import ...`), duplicate functions across files, wrong filenames, hardcoded CUDA calls, and 44MB of binary files tracked in git. This made the project unpresentable as a portfolio piece and unmaintainable for future development.

## Decision

Restructure into a proper Python package `mvpdr/` with:

- `mvpdr/clip/` — local CLIP copy (verbatim from OpenAI, no pip dependency)
- `mvpdr/datasets/` — consolidated dataset classes with shared `DatasetBase`
- `mvpdr/utils.py` — deduplicated utility functions (873 → ~130 lines)
- `mvpdr/models/` — new architecture modules (Phase 1)
- `train.py` / `train_plus.py` — entry points at repo root
- `pyproject.toml` — proper packaging with entry points

## Consequences

- **Positive:** Clean import paths (`from mvpdr.models import MVPDRPlus`), no circular dependencies, installable via pip
- **Positive:** Dead code eliminated (17+ duplicate functions removed), bugs fixed (stale logits, hardcoded CUDA, wrong tensor conversion)
- **Positive:** Binary files removed from git tracking via `.gitignore`
- **Negative:** Breaking change — old `src/` import paths no longer work
- **Negative:** Requires re-running any cached experiments (cache paths changed)
