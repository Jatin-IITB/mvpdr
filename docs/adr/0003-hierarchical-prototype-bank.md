# ADR-0003: Hierarchical Multi-Granularity Prototype Bank

**Status:** Accepted  
**Date:** 2026-08-07

## Context

The baseline MVPDR uses a single set of K-Means visual prototypes (e.g., 16 per class). This flat structure has limitations:

- A single granularity may be too coarse for fine-grained distinctions (early vs. late blight) or too fine for broad categories (healthy vs. diseased)
- No mechanism to adapt the prototype resolution based on input difficulty
- Prototypes are static after K-Means initialization — no refinement during training beyond linear adapter weight updates

## Decision

Implement `HierarchicalPrototypeBank` in `mvpdr/models/prototype_bank.py`:

- **Multi-granularity levels:** maintain prototypes at multiple scales (default: 4, 8, 16 clusters per class). Coarse prototypes capture broad disease categories; fine prototypes capture subtle visual variants.
- **Learned routing:** a lightweight MLP router (`D → D/4 → n_levels`) produces per-sample softmax weights over granularity levels, letting the model attend to coarse prototypes for easy samples and fine prototypes for ambiguous cases.
- **Learnable prototypes:** stored as `nn.Parameter` (not fixed buffers), initialized from K-Means and fine-tuned during training.
- **Per-level learnable beta:** the exponential affinity sharpness parameter `exp(-β(1-affinity))` is learned independently per level.
- **Momentum EMA update:** during training, prototypes drift toward matched training features via exponential moving average (β=0.999), preventing catastrophic forgetting of the K-Means initialization while allowing adaptation.

### Prototype Logit Computation (per level)

```
affinity_k = image @ protos_k.T          # [B, C*k]
logits_k = exp(-β_k(1 - affinity_k)) @ one_hot_k  # [B, C]
```

Final logits = Σ_k (router_weight_k * logits_k)

### Alternatives Considered

1. **Single-level with more clusters** — simpler but can't adapt granularity per sample. A 32-cluster bank wastes compute on easy samples.
2. **Mixture of Gaussians** — richer density model but much harder to train in few-shot settings.
3. **Slot Attention** — iterative prototype refinement per forward pass. Elegant but significantly more expensive and harder to stabilize.

## Consequences

- **Positive:** ~453K trainable params (for 27 classes, levels=[4,8,16]) — lightweight
- **Positive:** Router naturally learns difficulty-adaptive prototype selection
- **Positive:** Momentum updates prevent prototype drift while allowing gradual refinement
- **Positive:** `init_from_features()` provides strong K-Means initialization, avoiding random-init instability
- **Negative:** K-Means initialization requires an extra pass over training features at startup
- **Negative:** Momentum update adds overhead per training step (mitigated by `@torch.no_grad()`)
- **Negative:** In extreme few-shot (1-shot), some levels may have fewer samples than clusters — handled by padding/clamping
