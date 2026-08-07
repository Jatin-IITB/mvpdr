# ADR-0007: Phase 1 Audit Findings and Fixes

**Status:** Accepted  
**Date:** 2026-08-07

## Context

An independent audit agent reviewed all Phase 1 code before proceeding to Phase 2. The audit found 1 critical bug, 4 important issues, and 5 minor improvements.

## Findings and Resolutions

### Critical

| ID | Finding | Fix |
|----|---------|-----|
| C1 | `beta` parameter in `HierarchicalPrototypeBank` is unconstrained — can go negative under gradient updates, causing `exp(-beta*(1-affinity))` to overflow to Inf/NaN | Wrapped with `F.softplus(self.beta[level_idx])` to ensure beta is always positive |

### Important

| ID | Finding | Fix |
|----|---------|-----|
| I1 | Momentum update ran inside `forward()` before `backward()`/`optimizer.step()`, creating gradient-parameter staleness | Moved momentum update to `train_plus.py` after `optimizer.step()`. Removed `labels` param from `forward()` signature |
| I2 | Cross-attention operates entirely in float32 (2x memory vs fp16) | Accepted as intentional for training stability. Documented in I2 comment. Can add autocast later if memory-constrained |
| I3 | `pre_load_features()` returns labels as `.half()` — lossy for >2048 classes | Changed `utils.py` to return `labels.long()` |
| I4 | Textual auxiliary loss fires even when `use_cross_attn=False`, where it duplicates the main loss (effective weight 1+λ_t instead of 1) | Guarded with `if "textual_logits" in aux and model.use_cross_attn:` |

### Minor

| ID | Finding | Fix |
|----|---------|-----|
| M1 | 1-shot creates K identical duplicated prototypes | Added Gaussian noise (σ=0.01) to duplicated prototypes |
| M3 | `torch.load` without `weights_only=True` | Added `weights_only=True` |
| M4 | Joint `clip_grad_norm_` across all components | Changed to per-optimizer-group gradient clipping |
| M2 | Only logs first param group's LR | Accepted — all groups follow the same cosine schedule, just at different magnitudes |
| M5 | Identical configs across datasets | Accepted — intentional starting point, will tune per-dataset after baseline experiments |

### Verified Correct

- PromptLearner correctly replicates CLIP's `encode_text` pipeline
- CLIP backbone is truly frozen (zero gradients)
- Cross-attention pre-norm architecture with proper residuals
- K-Means initialization handles all edge cases (0, 1, <k samples)
- All ablation paths degenerate correctly
- `logit_scale` registered exactly once in optimizer

## Consequences

- All critical and important issues are resolved
- The model is safe against NaN from unbounded beta
- Momentum updates now happen at the correct point in the training loop
- Auxiliary loss behavior is correct for all component combinations
