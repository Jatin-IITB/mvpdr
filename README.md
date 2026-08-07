# MVPDR+: Multi-View Prototype-based Disease Recognition

AI-powered plant disease diagnostic system combining CLIP-based few-shot visual recognition, agentic LLM reasoning, and a structured treatment knowledge base.

**88.3% accuracy on PlantDoc with only 20 labeled samples per class.**

## Highlights

- **Few-shot learning** — CoOp learnable prompts + hierarchical prototype bank + cross-attention fusion on frozen CLIP
- **Visual explanations** — GradCAM heatmaps showing which leaf regions drive predictions
- **Label-free severity estimation** — estimates disease severity from GradCAM coverage and confidence signals without severity annotations
- **Agentic diagnostics** — Qwen 3 8B via Ollama reasons over classification results, severity evidence, and a 25-disease knowledge base to produce structured treatment reports
- **Open-set detection** — flags unknown diseases instead of misclassifying them (AUROC 0.94)
- **Domain adaptation** — transfers across geographies via CORAL + DANN (+5.2% on target domain)
- **Calibrated confidence** — temperature scaling reduces calibration error by 78%

## Quick Start

```bash
# Clone and install
git clone https://github.com/jatin-gupta/mvpdr.git
cd mvpdr
pip install -e .

# For agentic diagnostics (local LLM, free)
pip install ollama
ollama pull qwen3:8b

# Launch the demo
python app.py --config configs/plantdoc_plus.yaml --zero_shot
# Open http://localhost:7860
```

### Docker (one command)

```bash
docker compose up
# Demo at http://localhost:7860, Ollama at http://localhost:11434
```

## Results

### Classification Accuracy (% top-1, 20-shot)

| Method | PlantDoc (27 cls) | PlantVillage (38 cls) | PlantWild (15 cls) |
|--------|------------------:|----------------------:|-------------------:|
| CLIP Zero-Shot | 72.4 | 89.1 | 51.2 |
| MVPDR (baseline) | 82.1 | 95.2 | 68.4 |
| **MVPDR+ (ours)** | **88.3** | **97.1** | **72.3** |

### Component Ablation (PlantDoc, 20-shot)

| Component | Accuracy | Delta |
|-----------|--------:|------:|
| CLIP Zero-Shot | 72.4 | — |
| + CoOp Prompts | 78.1 | +5.7 |
| + Prototype Bank | 75.3 | +2.9 |
| + Cross-Attention | 74.8 | +2.4 |
| Full MVPDR+ | 88.3 | +15.9 |

### Additional Metrics

| Metric | Value |
|--------|------:|
| Open-set AUROC (Energy) | 0.941 |
| ECE (after calibration) | 0.031 |
| Domain adaptation gain | +5.2% |
| Severity expert agreement | 83% |
| Diagnostic latency (CPU) | ~3.2s |

## Architecture

```
Image ─→ CLIP Visual Encoder (frozen) ─→ Image Features [B, 512]
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ↓                         ↓                         ↓
             CoOp Prompt               Hierarchical              Cross-Attention
             Learner                   Prototype Bank            Fusion (2 layers)
             (learnable                (K-Means at               img Q, proto KV
              soft prompts)            4/8/16 scales)            8 heads
                    ↓                         ↓                         ↓
             Text Features [C,512]     Visual Logits [B,C]       Fused Logits [B,C]
                    │                         │                         │
                    └─────────────────────────┴─────────────────────────┘
                                              ↓
                                    Classification + Aux Losses
                                              ↓
                    ┌─────────────────────────┼──────────────────┐
                    ↓                         ↓                  ↓
              GradCAM                  Severity              Agentic LLM
              Heatmap                  Estimation            Pipeline (Qwen 3)
                                       (heuristic)                ↓
                                              ↓            Knowledge Base
                                    Structured Diagnostic Report
```

## Modules

### Core Model (`mvpdr/models/`)
- **PromptLearner** — CoOp continuous prompt embeddings (4-16 context tokens)
- **HierarchicalPrototypeBank** — multi-granularity K-Means with learned routing
- **CrossAttentionFusion** — transformer decoder fusing image features with prototypes
- **MVPDRPlus** — full model with toggleable components for ablation

### Analysis (`mvpdr/`)
- **interpretability.py** — GradCAM for ResNet and ViT CLIP backbones
- **openset.py** — MSP, Energy, Mahalanobis open-set scoring
- **adaptation.py** — CORAL covariance alignment + DANN gradient reversal
- **calibration.py** — temperature scaling + ECE metrics + reliability diagrams
- **severity.py** — label-free heuristic severity + CORAL ordinal head
- **knowledge.py** — 25-disease treatment knowledge base with fuzzy matching
- **agent.py** — agentic diagnostic pipeline (Ollama/Qwen → Anthropic → rule-based)

### Scripts
- `train.py` / `train_plus.py` — training for baseline / MVPDRPlus
- `train_domain_adapt.py` — domain adaptation training
- `scripts/diagnose.py` — full diagnostic pipeline CLI
- `scripts/calibrate.py` — post-hoc temperature scaling
- `scripts/evaluate_openset.py` — open-set detection evaluation
- `scripts/generate_figures.py` — publication-quality figures (300 DPI)
- `app.py` — Gradio web demo

## Training

```bash
# Baseline MVPDR
python train.py --config configs/plantdoc.yaml

# MVPDRPlus (CoOp + prototypes + cross-attention)
python train_plus.py --config configs/plantdoc_plus.yaml

# Domain adaptation (source → target)
python train_domain_adapt.py --config configs/plantdoc_plus.yaml \
    --target_config configs/plantwild_plus.yaml \
    --checkpoint results/.../model.pth
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_ctx` | 4 | CoOp context length |
| `prototype_levels` | (4, 8, 16) | K-Means cluster counts per level |
| `n_cross_layers` | 2 | Cross-attention depth |
| `n_heads` | 8 | Attention heads |
| `backbone` | ViT-B/16 | CLIP backbone |

## Diagnostics

```bash
# Full diagnostic report (rule-based, no LLM needed)
python scripts/diagnose.py --config configs/plantdoc_plus.yaml \
    --image path/to/leaf.jpg --output report.json

# With agentic reasoning (requires Ollama + qwen3:8b)
MVPDR_AGENT_BACKEND=ollama python scripts/diagnose.py \
    --config configs/plantdoc_plus.yaml \
    --image path/to/leaf.jpg
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MVPDR_AGENT_BACKEND` | auto | `ollama`, `anthropic`, or `rule` |
| `OLLAMA_MODEL` | `qwen3:8b` | Ollama model name |
| `OLLAMA_HOST` | `localhost:11434` | Ollama server URL |
| `ANTHROPIC_API_KEY` | — | Optional cloud fallback |

## Project Structure

```
mvpdr/
├── app.py                          # Gradio web demo
├── train.py / train_plus.py        # Training scripts
├── train_domain_adapt.py           # Domain adaptation
├── predict.py                      # Batch inference
├── docker-compose.yml              # One-command deployment
├── Dockerfile                      # Container image
├── configs/                        # YAML experiment configs
├── docs/
│   ├── adr/                        # 13 architectural decision records
│   └── RESUME_BRIEF.md             # Detailed project brief
├── figures/                        # Publication figures (300 DPI)
├── mvpdr/
│   ├── clip/                       # CLIP backbone (frozen)
│   ├── datasets/                   # PlantDoc, PlantVillage, PlantWild
│   ├── models/                     # MVPDRPlus components
│   │   ├── prompt_learner.py       # CoOp
│   │   ├── prototype_bank.py       # Hierarchical prototypes
│   │   ├── cross_attention.py      # Transformer fusion
│   │   └── mvpdr_model.py          # Full model
│   ├── data/
│   │   └── disease_knowledge.json  # 25-disease treatment KB
│   ├── agent.py                    # Agentic diagnostic pipeline
│   ├── severity.py                 # Severity estimation
│   ├── knowledge.py                # KB API + fuzzy matching
│   ├── interpretability.py         # GradCAM
│   ├── openset.py                  # Open-set detection
│   ├── adaptation.py               # CORAL + DANN
│   ├── calibration.py              # Temperature scaling
│   └── utils.py                    # Shared utilities
└── scripts/                        # Evaluation & visualization
```

## Citation

```bibtex
@misc{gupta2026mvpdr,
  title={MVPDR+: Multi-View Prototype-based Disease Recognition with Agentic Diagnostics},
  author={Gupta, Jatin},
  year={2026}
}
```

## License

MIT
