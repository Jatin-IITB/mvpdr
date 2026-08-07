# ADR-0012: Phase 4 — Agentic Diagnostic Pipeline

**Status:** Accepted  
**Date:** 2026-08-07

## Context

Phases 1-3 delivered a strong CLIP-based classification system with interpretability, open-set detection, domain adaptation, calibration, and a Gradio demo. Phase 4 elevates MVPDR+ from a classifier into a full **AI-powered diagnostic system** — the differentiator that makes this a flagship portfolio project.

The user explicitly requested: "latest tech stack, agentic AI stuff, ML" — something worthy of showcasing on a resume.

## Decisions

### D1: Heuristic Severity Estimation (No Severity Labels Required)

**Problem:** Plant disease datasets (PlantDoc, PlantVillage, PlantWild) lack severity annotations. Training an ordinal regression head requires labeled data we don't have.

**Solution:** A novel label-free severity estimator that combines three signals:
1. **GradCAM spatial coverage** — ratio of activated pixels above threshold. Larger diseased area → higher severity.
2. **Classification confidence** — high confidence in a disease class correlates with clear, advanced symptoms.
3. **Feature distance from healthy prototype** — how far the image embedding is from the healthy-class centroid.

These three signals are combined via a configurable weighted score into a 4-level severity scale: Healthy / Mild / Moderate / Severe.

**Additionally:** A trainable `OrdinalSeverityHead` using CORAL ordinal regression (Cao et al. 2020) is included for when labeled severity data becomes available.

**Rationale:** This is a genuine methodological contribution — using model interpretability signals as proxy severity indicators without requiring additional annotations. Novel and publishable.

### D2: Structured Disease Knowledge Base

**Problem:** Classification alone doesn't help farmers. They need actionable treatment recommendations.

**Solution:** A JSON knowledge base (`data/disease_knowledge.json`) with structured entries per disease:
- Scientific name, pathogen type (fungal/bacterial/viral)
- Symptoms at each severity level
- Treatment protocols (chemical, biological, cultural)
- Prevention strategies
- Environmental risk factors

A Python API (`mvpdr/knowledge.py`) provides lookup, fuzzy matching, and CLIP-based semantic retrieval for unknown disease names.

### D3: Agentic Diagnostic Pipeline with Local LLM (Ollama + Qwen 3)

**Problem:** Static model outputs don't provide the contextual reasoning that makes a diagnosis actionable.

**Solution:** An agentic pipeline (`mvpdr/agent.py`) using **Ollama** with **Qwen 3 8B** as the default local LLM backend. The agent uses tool-calling to chain MVPDR+ inference with knowledge base retrieval.

**Architecture:**
```
Image → MVPDR+ Classification → Severity Estimation → GradCAM
                    ↓
         LLM Agent (Ollama/Qwen 3 8B — local, free)
         ├── Tool: get_classification_results — MVPDR+ predictions
         ├── Tool: get_severity_assessment — severity level + evidence
         ├── Tool: lookup_disease_info — query knowledge base
         ├── Tool: get_treatment_protocol — retrieve treatment protocol
         └── Tool: submit_diagnostic_report — structured output
                    ↓
         JSON Report (diagnosis, severity, treatment, confidence, caveats)
```

The agent operates in a **multi-step reasoning loop**:
1. Examines classification results and confidence scores
2. Considers severity evidence (GradCAM coverage, feature distances)
3. Cross-references the knowledge base for treatments
4. Produces a structured diagnostic report with reasoning chain

**Backend priority:**
1. **Ollama** (default) — local, free, no API key. Uses Qwen 3 8B which has strong tool-calling support. Override model with `OLLAMA_MODEL` env var.
2. **Anthropic** — cloud fallback if `ANTHROPIC_API_KEY` is set and Ollama unavailable.
3. **Rule-based** — deterministic fallback if no LLM is available.

Force a specific backend with `MVPDR_AGENT_BACKEND=ollama|anthropic|rule`.

**Why Qwen 3 8B:** Excellent tool-calling accuracy, runs locally on consumer hardware (8GB VRAM or CPU), Apache 2.0 license, state-of-the-art for its size class.

### D4: Structured Diagnostic Report Schema

Reports follow a Pydantic-validated schema:
- `disease_name`, `confidence`, `severity_level`, `severity_evidence`
- `affected_area_percent` (from GradCAM)
- `treatment_plan` (immediate + long-term)
- `differential_diagnosis` (alternative possibilities)
- `reasoning_chain` (agent's step-by-step logic)
- `caveats` (limitations, when to consult an expert)

### D5: Updated Gradio Demo

The demo gains a new "AI Diagnostic Report" tab that shows the full agentic analysis alongside existing classification/GradCAM outputs. Works in both agentic mode (with API key) and fallback mode.

## Consequences

- MVPDR+ becomes a complete diagnostic system, not just a classifier
- The agentic pipeline demonstrates modern AI engineering (tool-use, structured output, graceful degradation)
- Severity estimation works without additional annotations — novel contribution
- Knowledge base is extensible for new diseases/crops
- Portfolio-worthy: combines CV + NLP + agentic AI + knowledge engineering

## Dependencies Added

- `ollama>=0.4.0` (local LLM agent — default backend)
- `pydantic>=2.0` (structured report validation)
- `anthropic>=1.0` (optional cloud fallback — `pip install mvpdr[cloud-agent]`)
