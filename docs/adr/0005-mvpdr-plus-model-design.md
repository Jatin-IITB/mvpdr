# ADR-0005: MVPDRPlus Model Design with Config-Driven Toggles

**Status:** Accepted  
**Date:** 2026-08-07

## Context

Phase 1 introduces three independent architectural upgrades (CoOp prompts, hierarchical prototypes, cross-attention fusion). For rigorous evaluation, we need to ablate each component independently and in combination to measure their individual and joint contributions.

## Decision

Implement `MVPDRPlus` as a single model class with config-driven component toggles:

```python
config = {
    "use_prompt_learner": True,   # CoOp vs fixed CLIP text
    "use_prototype_bank": True,   # Hierarchical vs flat prototypes
    "use_cross_attn": True,       # Cross-attention vs weighted sum
}
model = MVPDRPlus(clip_model, classnames, config)
```

### Forward Pass Design

```
image_features [B, D]  (from frozen CLIP)
       │
       ├─→ PromptLearner(clip_model) → text_features [C, D]
       │         └─→ textual_logits = image @ text.T
       │
       ├─→ PrototypeBank(image) → visual_logits [B, C]
       │
       └─→ CrossAttentionFusion(image, [text_protos ∥ visual_protos])
                  └─→ fused = normalize(cross_attn_output)
                  └─→ logits = scale * fused @ text.T

Returns: (logits, aux_dict)
```

### Training Strategy

- **Auxiliary losses:** `loss = CE(fused) + λ_v * CE(visual) + λ_t * CE(textual)` — each branch is trained to be independently discriminative
- **Per-component learning rates:** prompt (2e-3), prototypes (1e-3), fusion (5e-4) — components closer to the frozen backbone get higher LR
- **CLIP frozen:** `requires_grad=False` on all CLIP params; gradients flow through for prompt learner but accumulate nowhere in CLIP
- **Momentum update:** prototype bank EMA runs inside `forward()` when `training=True` and labels provided

### Fallback Behavior

When components are disabled:
- No prompt_learner → caller must provide `text_features` argument
- No prototype_bank → no `visual_logits` in aux, no momentum updates
- No cross_attn → falls back to `logits = t_logits + alpha * v_logits` (baseline behavior)

## Consequences

- **Positive:** Full 2³ = 8 ablation grid from a single model class
- **Positive:** Separate `train.py` (baseline) and `train_plus.py` (upgraded) coexist without conflict
- **Positive:** Model checkpoint includes config, enabling reproducible loading
- **Negative:** `forward()` signature has 4 args (image_features, clip_model, text_features, labels) — more complex than the baseline's
- **Negative:** Config dict is untyped — invalid keys silently ignored (mitigated by sensible defaults)
