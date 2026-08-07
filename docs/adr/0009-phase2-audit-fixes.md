# ADR-0009: Phase 2 Audit Findings and Fixes

**Status:** Accepted  
**Date:** 2026-08-07

## Context

An independent audit agent reviewed all Phase 2 code before proceeding to Phase 3. The audit found 2 critical bugs, 3 important issues, and 2 minor observations.

## Findings and Resolutions

### Critical

| ID | Finding | Fix |
|----|---------|-----|
| C2 | `interpretability.py`: null check on `self._feature_map` happens AFTER accessing `.grad`, causing `AttributeError` if hooks didn't fire | Split into two checks — first `fm is None`, then `grad is None` — each with a descriptive error message |
| C3 | `train_domain_adapt.py`: CORAL/DANN computed on raw CLIP features (extracted inside `torch.no_grad()`), so DA loss produces zero gradient to learnable components | Run target features through MVPDRPlus too, apply CORAL/DANN on model output logits so gradients flow back through prompt learner / prototype bank / cross-attention. Updated DomainDiscriminator input_dim to `n_classes` |

### Important

| ID | Finding | Fix |
|----|---------|-----|
| I1 | `evaluate_openset.py`: `fit_mahalanobis` called with `n_classes` (total) but only known-class labels present — unknown-class slots get zero-mean centroids that corrupt scoring | Remap known labels to contiguous `[0, n_known)` range before fitting, pass `len(known_classes)` |
| I2 | `calibrate.py`: division by zero when computing ECE improvement percentage if `ece_before == 0` | Guard with `max(ece_before, 1e-8)` |
| I3 | `interpretability.py`: ViT GradCAM slices LND-format tensors assuming batch_size=1, would produce wrong spatial grid for larger batches | Added assertion `fm.shape[1] == 1` with clear error message |

### Minor (accepted)

| ID | Finding | Status |
|----|---------|--------|
| M1 | `train_domain_adapt.py` saves checkpoints without `weights_only` flag | N/A — `weights_only` is a load-side concern, not save-side |
| M2 | ECE first bin excludes confidence = 0.0 exactly | Benign — softmax never produces exactly 0 |

### Verified Correct

- GradCAM `retain_grad()` approach works with frozen CLIP — `@torch.enable_grad()` ensures computation graph is built, gradients flow through activations (not params) back to the input image
- CORAL covariance uses `max(n-1, 1)` for single-sample safety
- Energy and MSP scoring handle temperature correctly
- Temperature scaling LBFGS closure correctly detaches inputs
- Gradient reversal layer negates gradients correctly (verified with unit test)
- All entry-point scripts use `torch.load(weights_only=True)`
- GRL alpha schedule ramps 0 → 1 correctly

## Consequences

- Domain adaptation now actually adapts the model (gradients reach learnable components)
- Mahalanobis scoring no longer corrupted by dummy centroids for unknown classes
- GradCAM fails gracefully with clear messages instead of cryptic AttributeErrors
