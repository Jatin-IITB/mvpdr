# ADR-0002: CoOp-Style Learnable Soft Prompts

**Status:** Accepted  
**Date:** 2026-08-07

## Context

The baseline MVPDR uses fixed text templates ("a photo of a {class}") and GPT-generated disease descriptions passed through CLIP's frozen text encoder. These fixed prompts may not capture the optimal textual representation for downstream few-shot classification — the prompt phrasing significantly affects CLIP's zero-shot performance (up to 15% variance across templates per the CLIP paper).

CoOp (Context Optimization, Zhou et al. 2022) showed that replacing discrete prompt tokens with learnable continuous vectors and optimizing them end-to-end yields consistent gains across vision-language tasks.

## Decision

Implement `PromptLearner` in `mvpdr/models/prompt_learner.py`:

- Replace positions 1..m in the CLIP token sequence with m learnable vectors
- Per-class prompt structure: `[SOS] [v1..vm] [class_name_tokens] [EOS] [PAD]`
- Support both unified context (shared across classes, 2,048 params for m=4) and class-specific context
- Optional initialization from a text string (e.g., "a photo of a") for warm-starting
- Forward pass: construct embeddings → add positional encoding → pass through CLIP's frozen text transformer → extract EOS features → project via text_projection
- Gradients flow through frozen transformer operations back to learnable vectors; CLIP parameters accumulate no gradients (`requires_grad=False`)

### Alternatives Considered

1. **CoCoOp (conditional context)** — generates per-image context via a meta-net. More powerful but adds complexity and latency. Deferred to a future phase.
2. **ProDA (prompt distribution)** — models prompt distribution rather than point estimates. Requires ensembling at inference. Too complex for initial implementation.
3. **MaPLe (multi-modal prompts)** — injects learnable tokens into both text and vision transformers. Requires modifying the vision encoder, breaking our "frozen backbone" constraint.

## Consequences

- **Positive:** Dataset-adaptive text representations learned end-to-end with only 2,048 extra parameters (m=4, dim=512)
- **Positive:** Seamlessly integrates with cross-attention fusion — learned text features serve as both classification anchors and attention keys
- **Positive:** Ablation-friendly — can be disabled via config to isolate its contribution
- **Negative:** Adds a dependency on CLIP's internal architecture (token_embedding, transformer, ln_final, text_projection) — any CLIP model change would require adaptation
- **Negative:** Requires careful dtype handling (CLIP operates in fp16, learnable params in fp16 matching CLIP's convention)
