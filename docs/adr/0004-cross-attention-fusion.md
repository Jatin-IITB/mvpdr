# ADR-0004: Cross-Attention Fusion Replacing Weighted Sum

**Status:** Accepted  
**Date:** 2026-08-07

## Context

The baseline fuses visual and textual logits via a fixed weighted sum:

```python
logits = textual_logits + alpha * visual_logits
```

where `alpha` is a scalar hyperparameter. This has two fundamental limitations:

1. **Sample-agnostic:** the same fusion weight applies to every input, regardless of whether the visual or textual view is more informative for that particular image
2. **Late fusion:** logits are combined after independent classification, losing the opportunity for cross-view feature refinement before the classification decision

## Decision

Implement `CrossAttentionFusion` in `mvpdr/models/cross_attention.py`:

- **Architecture:** stack of L pre-norm cross-attention layers (default L=2), each containing:
  - LayerNorm → MultiheadAttention(Q=image, KV=prototypes) → residual
  - LayerNorm → FFN(D→4D→D with GELU) → residual
  - Final LayerNorm on output
- **Input:** image features [B, D] as query; concatenated textual + visual prototypes [N, D] as key/value
- **Output:** fused features [B, D] + attention weights [B, N] for interpretability
- **Classification:** fused features classified via cosine similarity to text class centroids, scaled by a learnable logit_scale: `logit_scale.exp() * fused @ text_features.T`
- **Computation in float32** for training stability (inputs cast from CLIP's fp16)
- 8 attention heads, matching CLIP's text transformer head count for the 512-dim embedding

### Why Cross-Attention Over Self-Attention

Self-attention over a concatenated [image; prototypes] sequence would let prototypes attend to each other, which is unnecessary — they're fixed reference points. Cross-attention is more parameter-efficient: the image feature is the only query, attending over prototypes as a key-value bank.

### Alternatives Considered

1. **Gated fusion:** `gate = σ(W[v_logits; t_logits])`, `logits = gate * v + (1-gate) * t`. Sample-adaptive but still late fusion (operates on logits, not features).
2. **FiLM conditioning:** textual features modulate visual features via affine transform. Cheaper but less expressive than attention.
3. **Transformer decoder with multiple queries:** treat each class prototype as a separate query. Richer but O(C*N) cost and harder to train in few-shot.

## Consequences

- **Positive:** Sample-adaptive fusion — the model learns to weight visual vs. textual prototypes per input via attention
- **Positive:** Early fusion — cross-view information exchange happens at the feature level, before classification
- **Positive:** Attention weights are interpretable — can visualize which prototypes each image attends to
- **Positive:** Returns both fused logits AND per-branch auxiliary logits for multi-task training
- **Negative:** Largest component by parameter count (~6.3M for 2 layers, 512-dim). Dominates the parameter budget.
- **Negative:** Quadratic in prototype count (N prototypes = N_text + N_visual). With levels=[4,8,16] and 38 classes, N ≈ 38 + 38*28 = 1,102 — manageable but worth monitoring.
