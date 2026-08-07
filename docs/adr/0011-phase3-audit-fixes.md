# ADR-0011: Phase 3 Audit Findings and Fixes

**Status:** Accepted  
**Date:** 2026-08-07

## Context

An independent audit reviewed the Gradio demo (`app.py`) and figure generator (`scripts/generate_figures.py`).

## Findings and Resolutions

### Important

| ID | Finding | Fix |
|----|---------|-----|
| I1 | Concurrent Gradio requests overwrite shared temp files (`mvpdr_conf.png`, `mvpdr_cam.png`) | Added PID suffix to temp file paths for process-level uniqueness |

### Minor

| ID | Finding | Fix |
|----|---------|-----|
| M1 | Both `run_btn.click` and `image_input.change` trigger prediction — double work on upload | Removed `.change` binding; classify only on button click |
| M3 | Method comparison y-axis goes to 105% (above 100% is nonsensical) | Changed to `set_ylim(40, 100)` |

### Verified Correct

- GradCAM text_features: both MVPDRPlus and zero-shot paths produce correctly normalized `[C, D]` features — prompt_learner output IS already normalized via `F.normalize` on line 155
- GradCAM runs its own forward pass with `@torch.enable_grad()` internally, so the outer `torch.no_grad()` block does not interfere
- Zero-shot mode correctly uses `clip_classifier().t()` to get `[C, D]` text features
- Global `_state` is write-once at startup, safe for concurrent reads

## Consequences

- Demo is safe for concurrent requests (no file collision)
- Cleaner UX with single classify button
- Figures have correct axis bounds
