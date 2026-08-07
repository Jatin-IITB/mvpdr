# ADR-0010: Phase 3 — Gradio Demo and Publication Figures

**Status:** Accepted  
**Date:** 2026-08-07

## Context

Phases 1-2 built the model and supporting capabilities. Phase 3 makes the project presentable as a portfolio piece: an interactive demo for live exploration and a figure-generation pipeline for paper/poster use.

## Decisions

### 1. Gradio Interactive Demo (`app.py`)

**Approach:** Single-file Gradio app that loads MVPDRPlus (or baseline) from a checkpoint and provides:
- Image upload with drag-and-drop
- Top-k prediction bar chart with confidence scores
- GradCAM heatmap overlay showing which regions drive the prediction
- Prototype attention visualization (when cross-attention is enabled)
- Open-set confidence score (Energy score) for "is this a known disease?" detection

**Design choices:**
- Loads model once at startup, caches CLIP backbone — avoids re-downloading per request
- Works without GPU (CPU fallback) for demo portability
- Accepts both MVPDRPlus checkpoints (`mvpdr_plus_model.pth`) and raw state dicts
- GradCAM runs on-demand per image (not pre-computed)

**Alternatives rejected:**
- Streamlit: heavier, requires separate server process, less ML-focused
- Flask + custom frontend: more effort, no built-in image handling or UI components

### 2. Publication Figure Generator (`scripts/generate_figures.py`)

**Figures produced:**
- Architecture overview diagram (matplotlib-drawn, not external tool)
- Few-shot accuracy curves (0/1/5/10/20 shot) across datasets
- Component ablation chart (with/without each Phase 1 component)
- Method comparison bar chart (CLIP zero-shot vs MVPDR vs MVPDR+)

All figures use consistent matplotlib styling (serif fonts, 300 DPI, tight layout) suitable for academic papers.

## Consequences

- **Positive:** Portfolio-ready demo that can be shared as a link (Gradio supports Hugging Face Spaces deployment)
- **Positive:** Figures are reproducible from saved results JSON files
- **Negative:** Gradio adds a heavyweight dependency (~100MB with all transitive deps)
