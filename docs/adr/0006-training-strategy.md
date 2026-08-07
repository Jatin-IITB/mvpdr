# ADR-0006: Per-Component Learning Rates and Auxiliary Losses

**Status:** Accepted  
**Date:** 2026-08-07

## Context

The upgraded MVPDR+ model has three components with very different characteristics:

| Component       | Parameters | Nature                          |
|-----------------|------------|----------------------------------|
| PromptLearner   | ~2K        | Operates in CLIP's embedding space, must stay near pre-trained manifold |
| PrototypeBank   | ~450K      | Initialized from K-Means, needs careful fine-tuning to preserve cluster structure |
| CrossAttention  | ~6.3M      | Randomly initialized, needs standard training from scratch |

A single learning rate would either under-train the fusion layers or over-train the prompt vectors.

## Decision

### Per-Component Learning Rates

```yaml
lr_prompt: 0.002    # High — small param count, needs strong signal
lr_proto:  0.001    # Medium — K-Means init provides good starting point
lr_fusion: 0.0005   # Lower — large param count, random init, needs stable convergence
```

All groups share a single cosine annealing schedule (CosineAnnealingLR) over `train_epoch * len(train_loader)` total steps, with AdamW optimizer (weight_decay=0.01).

### Auxiliary Loss Design

```python
loss = CE(fused_logits, target)
     + lambda_v * CE(visual_logits, target)      # default 0.5
     + lambda_t * CE(textual_logits, target)      # default 0.5
```

**Rationale:** Without auxiliary losses, the fusion module could learn to ignore one branch entirely (e.g., setting attention weights to zero for all visual prototypes). The per-branch CE losses ensure each view remains independently discriminative, which:

1. Provides richer gradient signal to the prototype bank and prompt learner
2. Enables meaningful ablation — each branch should work standalone
3. Acts as implicit regularization against fusion-layer overfitting

### Alternatives Considered

1. **LARS/LAMB optimizer** — per-layer adaptive LR. More principled but adds complexity and tuning. Standard AdamW with manual LR groups is simpler and well-understood.
2. **Gradual unfreezing** — train fusion first, then add prompt, then protos. Adds complexity with unclear benefit for our model size.
3. **KL-divergence loss between branches** — forces visual and textual predictions to agree. Could over-constrain and prevent the branches from specializing.
4. **No auxiliary losses** — simpler but risks branch collapse where fusion ignores one view.

## Consequences

- **Positive:** Each component trains at its natural rate
- **Positive:** Auxiliary losses prevent branch collapse and improve per-branch standalone performance
- **Positive:** Single scheduler simplifies training — all components follow the same warmup/decay profile
- **Negative:** Three extra hyperparameters (lr_prompt, lr_proto, lr_fusion) + two loss weights (lambda_v, lambda_t)
- **Negative:** Auxiliary loss weights need tuning — current defaults (0.5, 0.5) are reasonable but dataset-dependent
