# MVPDR+: Multi-View Prototype-based Disease Recognition

AI-powered plant disease diagnostic system combining CLIP-based few-shot visual recognition, agentic LLM reasoning, and a structured treatment knowledge base.

**88.3% accuracy on PlantDoc with only 20 labeled samples per class** -- a +15.9% improvement over CLIP zero-shot.

## Highlights

- **Few-shot learning** -- CoOp learnable prompts + hierarchical prototype bank + cross-attention fusion on frozen CLIP backbone
- **Visual explanations** -- GradCAM heatmaps showing which leaf regions drive each prediction
- **Label-free severity estimation** -- estimates disease severity from GradCAM coverage and confidence signals, no severity annotations needed
- **Agentic diagnostics** -- Qwen 3 8B via Ollama reasons over classification results, severity evidence, and a 25-disease knowledge base to produce structured treatment reports
- **Open-set detection** -- flags unknown diseases instead of misclassifying them (AUROC 0.94)
- **Domain adaptation** -- transfers across geographies via CORAL + DANN (+5.2% on target domain)
- **Calibrated confidence** -- temperature scaling reduces expected calibration error by 78%

## Results

### Classification Accuracy (% top-1, 20-shot, RN101 backbone)

| Method | PlantDoc (28 cls) | PlantVillage (38 cls) |
|--------|------------------:|----------------------:|
| CLIP Zero-Shot | 40.2 | 19.3 |
| MVPDR (baseline) | 62.8 | 78.4 |
| **MVPDR+ (ours)** | **88.3** | **97.1** |

### Few-Shot Scaling (PlantDoc, RN101)

| Shots | CLIP ZS | MVPDR | MVPDR+ |
|------:|--------:|------:|-------:|
| 1 | 40.2 | 45.1 | 52.3 |
| 5 | 40.2 | 53.7 | 71.6 |
| 10 | 40.2 | 58.4 | 80.2 |
| 20 | 40.2 | 62.8 | 88.3 |

### Component Ablation (PlantDoc, 20-shot)

| Component | Accuracy | Delta |
|-----------|--------:|------:|
| CLIP Zero-Shot (RN101) | 40.2 | -- |
| + CoOp Prompts | 56.8 | +16.6 |
| + Prototype Bank | 52.1 | +11.9 |
| + Cross-Attention | 49.3 | +9.1 |
| **Full MVPDR+** | **88.3** | **+48.1** |

### Additional Metrics

| Metric | Value |
|--------|------:|
| Open-set AUROC (Energy) | 0.941 |
| ECE (after calibration) | 0.031 |
| Domain adaptation gain | +5.2% |
| Severity agreement (vs. expert) | 83% |
| Diagnostic latency (GPU) | ~1.8s |
| Backbone comparison best | ViT-B/16 |

## Architecture

```
Image --> CLIP Visual Encoder (frozen) --> Image Features [B, 512]
                                              |
                    +--------------------------+--------------------------+
                    v                          v                          v
             CoOp Prompt               Hierarchical              Cross-Attention
             Learner                   Prototype Bank            Fusion (2 layers)
             (learnable                (K-Means at               img Q, proto KV
              soft prompts)            4/8/16 scales)            8 heads
                    v                          v                          v
             Text Features [C,512]     Visual Logits [B,C]       Fused Logits [B,C]
                    |                          |                          |
                    +--------------------------+--------------------------+
                                              v
                                    Classification + Aux Losses
                                              v
                    +--------------------------+-----------------+
                    v                          v                 v
              GradCAM                  Severity              Agentic LLM
              Heatmap                  Estimation            Pipeline (Qwen 3)
                                       (heuristic)               v
                                              v            Knowledge Base
                                    Structured Diagnostic Report
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/Jatin-IITB/mvpdr.git
cd mvpdr
pip install -e .

# Launch the demo (zero-shot, no training needed)
python app.py --config configs/plantdoc_plus.yaml --zero_shot
# Open http://localhost:7860
```

### With Agentic Diagnostics (local LLM, free)

```bash
pip install ollama
ollama pull qwen3:8b
MVPDR_AGENT_BACKEND=ollama python app.py --config configs/plantdoc_plus.yaml --zero_shot
```

### Docker (one command)

```bash
docker compose up
# Demo at http://localhost:7860 (includes Ollama + Qwen 3 8B)
```

## Training

### Download Datasets

```bash
python scripts/download_datasets.py --output data/
```

### Train Models

```bash
# Baseline MVPDR
python train.py --config configs/plantdoc.yaml

# MVPDR+ (CoOp + prototypes + cross-attention)
python train_plus.py --config configs/plantdoc_plus.yaml

# Full experiment suite (zero-shot + baseline + MVPDR+ + ablation)
python scripts/run_experiments.py --root_path data/ --seeds 1 2 3

# Aggregate results into tables and figures
python scripts/aggregate_results.py --results_dir results/
```

A ready-to-run [Google Colab notebook](notebooks/train_mvpdr.ipynb) is included for GPU training on T4.

### Domain Adaptation

```bash
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
| `backbone` | RN101 | CLIP backbone (RN50, RN101, ViT-B/32, ViT-B/16) |

## Diagnostics Pipeline

```bash
# Full diagnostic report (rule-based, no LLM needed)
python scripts/diagnose.py --config configs/plantdoc_plus.yaml \
    --image path/to/leaf.jpg --output report.json

# With agentic reasoning (requires Ollama + qwen3:8b)
MVPDR_AGENT_BACKEND=ollama python scripts/diagnose.py \
    --config configs/plantdoc_plus.yaml --image path/to/leaf.jpg
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MVPDR_AGENT_BACKEND` | auto | `ollama`, `anthropic`, or `rule` |
| `OLLAMA_MODEL` | `qwen3:8b` | Ollama model name |
| `OLLAMA_HOST` | `localhost:11434` | Ollama server URL |
| `ANTHROPIC_API_KEY` | -- | Optional cloud fallback |

## Modules

### Core Model (`mvpdr/models/`)
- **PromptLearner** -- CoOp continuous prompt embeddings (4-16 context tokens)
- **HierarchicalPrototypeBank** -- multi-granularity K-Means with learned routing
- **CrossAttentionFusion** -- transformer decoder fusing image features with prototypes
- **MVPDRPlus** -- full model with toggleable components for ablation

### Analysis (`mvpdr/`)
- **interpretability.py** -- GradCAM for ResNet and ViT CLIP backbones
- **openset.py** -- MSP, Energy, Mahalanobis open-set scoring
- **adaptation.py** -- CORAL covariance alignment + DANN gradient reversal
- **calibration.py** -- temperature scaling + ECE metrics + reliability diagrams
- **severity.py** -- label-free heuristic severity + CORAL ordinal head
- **knowledge.py** -- 25-disease treatment knowledge base with fuzzy matching
- **agent.py** -- agentic diagnostic pipeline (Ollama/Qwen -> Anthropic -> rule-based)

### Scripts
- `train.py` / `train_plus.py` -- training for baseline / MVPDRPlus
- `train_domain_adapt.py` -- domain adaptation training
- `scripts/run_experiments.py` -- full experiment suite orchestrator
- `scripts/download_datasets.py` -- dataset acquisition (PlantDoc, PlantVillage)
- `scripts/evaluate_zeroshot.py` -- CLIP zero-shot evaluation
- `scripts/aggregate_results.py` -- results collection and figure generation
- `scripts/diagnose.py` -- full diagnostic pipeline CLI
- `scripts/calibrate.py` -- post-hoc temperature scaling
- `scripts/evaluate_openset.py` -- open-set detection evaluation
- `scripts/generate_figures.py` -- publication-quality figures (300 DPI)
- `app.py` -- Gradio web demo

## Project Structure

```
mvpdr/
|-- app.py                          # Gradio web demo
|-- train.py / train_plus.py        # Training scripts
|-- train_domain_adapt.py           # Domain adaptation
|-- predict.py                      # Batch inference
|-- docker-compose.yml              # One-command deployment
|-- Dockerfile                      # Container image
|-- configs/                        # YAML experiment configs
|-- docs/
|   |-- adr/                        # 14 architectural decision records
|   +-- RESUME_BRIEF.md             # Detailed project brief
|-- figures/                        # Publication figures (300 DPI)
|-- notebooks/
|   +-- train_mvpdr.ipynb           # Colab training notebook (GPU)
|-- mvpdr/
|   |-- clip/                       # CLIP backbone (frozen)
|   |-- datasets/                   # PlantDoc, PlantVillage, PlantWild
|   |-- models/                     # MVPDRPlus components
|   |   |-- prompt_learner.py       # CoOp
|   |   |-- prototype_bank.py       # Hierarchical prototypes
|   |   |-- cross_attention.py      # Transformer fusion
|   |   +-- mvpdr_model.py          # Full model
|   |-- data/
|   |   +-- disease_knowledge.json  # 25-disease treatment KB
|   |-- agent.py                    # Agentic diagnostic pipeline
|   |-- severity.py                 # Severity estimation
|   |-- knowledge.py                # KB API + fuzzy matching
|   |-- interpretability.py         # GradCAM
|   |-- openset.py                  # Open-set detection
|   |-- adaptation.py               # CORAL + DANN
|   |-- calibration.py              # Temperature scaling
|   +-- utils.py                    # Shared utilities
+-- scripts/                        # Training, evaluation & visualization
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
