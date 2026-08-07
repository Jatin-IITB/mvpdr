# ADR-0013: Phase 4 Audit Findings and Fixes

**Status:** Accepted  
**Date:** 2026-08-07

## Context

An independent audit reviewed Phase 4 (agentic diagnostic pipeline, severity estimation, knowledge base). Additionally, the agentic backend was switched from Anthropic-only to Ollama/Qwen 3 as primary (local, free) with Anthropic as optional cloud fallback.

## Findings and Resolutions

### Critical

| ID | Finding | Fix |
|----|---------|-----|
| C1 | Agent loop returns immediately on `submit_diagnostic_report`, dropping other batched tool calls in the same response | Process all tool calls first, collect submit_input, return report only after full batch is handled |
| C2 | `scripts/diagnose.py` calls `model.prompt_learner(clip_model)` without None check — crashes when `use_prompt_learner: False` | Added guard: falls back to `clip_classifier()` when prompt_learner is None |
| C3 | Whitespace-only `ANTHROPIC_API_KEY` passes truthiness check → opaque 401 from API; no exception handling around API loop | Added `.strip()` on key check; wrapped entire agent loop in try/except with fallback to rule-based |

### Important

| ID | Finding | Fix |
|----|---------|-----|
| I3 | GradCAM map passed at different resolutions: CLI uses raw ~7x7 cam, Gradio uses full-image-resized cam → different severity scores for same image | CLI now resizes cam to image dimensions before severity estimation, matching Gradio behavior |
| I5 | PID-based temp file names in Gradio cause collisions under multi-worker deployment (same PID in forked workers) | Switched to `uuid4().hex[:8]` suffix for per-request uniqueness |
| I7 | CORAL ordinal head `predict_probs()` can produce negative probabilities early in training due to unconstrained biases | Added `.clamp(min=0.0)` to output probabilities |

### Design Change

| ID | Change | Rationale |
|----|--------|-----------|
| D1 | Switched default agent backend from Anthropic to Ollama + Qwen 3 8B | Local execution (free, no API key), Qwen 3 8B has strong tool-calling, self-contained project |
| D2 | Added multi-backend support: `MVPDR_AGENT_BACKEND=ollama\|anthropic\|rule` | Flexibility for different deployment scenarios |

### Verified Correct

- Knowledge base fuzzy matching resolves all 25 disease aliases correctly
- Tool handler JSON serialization is safe (no user input reaches shell or file paths)
- Rule-based fallback produces complete reports with treatments when KB is available
- OpenAI and Anthropic tool format definitions are structurally equivalent
- `_parse_agent_report` handles missing/malformed fields gracefully with defaults

## Consequences

- Agent pipeline works out-of-the-box with `ollama pull qwen3:8b` — no API key
- Consistent severity scores across CLI and Gradio entry points
- Concurrent Gradio requests no longer collide on temp files
- Anthropic API errors degrade gracefully to rule-based reports
