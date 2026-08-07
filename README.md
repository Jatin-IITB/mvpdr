# MVPDR: Multi-View Prototype-based Disease Recognition

CLIP-based plant disease classification using learnable dual-adapter architecture with visual and textual prototypes. Achieves strong few-shot and zero-shot performance across multiple plant disease benchmarks with minimal labeled data.

## Key Results

| Dataset     | Classes | 0-shot | 1-shot | 5-shot | 10-shot | 20-shot |
|-------------|---------|--------|--------|--------|---------|---------|
| PlantDoc    | 27      | 72.36% | —      | —      | —       | 85.26%  |
| PlantVillage| 38      | —      | —      | —      | —       | —       |
| PlantWild   | 18      | —      | —      | —      | —       | —       |

## Method

MVPDR uses a frozen CLIP backbone with two learnable linear adapters:

1. **Visual Adapter** — maps image features to K-Means cluster prototypes per class, producing visual logits via exponential affinity scoring
2. **Textual Adapter** — maps image features to GPT-generated disease description embeddings, combining mean and max pooling over prompts

Final prediction fuses both views: `logits = textual_logits + alpha * visual_logits`

## Setup

```bash
pip install -r requirements.txt
```

Datasets should be placed under a root directory with structure:
```
<root_path>/plantdoc/images/<class_name>/<images>
<root_path>/plantvillage/images/<class_name>/<images>
<root_path>/plantwild/images/<class_name>/<images>
```

## Training

```bash
# Zero-shot on PlantDoc
python train.py --config configs/plantdoc.yaml

# 5-shot with specific hyperparameters
python train.py --config configs/plantdoc_5shot.yaml --alpha 0.3 --nclt 16 --seed 1

# Different backbone
python train.py --config configs/plantdoc.yaml --backbone ViT-B/32

# Use cached features (after first run)
python train.py --config configs/plantdoc.yaml  # set load_cache/load_pre_feat: true in yaml
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha` | 0.3     | Visual-textual fusion weight |
| `--bbeta` | 0.5     | Prototype affinity temperature |
| `--gamma` | 0.5     | Textual mean/max fusion balance |
| `--nclt`  | 16      | Visual prototype clusters per class |
| `--backbone` | RN101 | CLIP backbone (RN50, RN101, ViT-B/32, ViT-B/16) |

## Inference

```bash
python predict.py --model results/plantdoc/RN101/.../mvpdr_model.pth \
                  --image path/to/leaf.jpg \
                  --classes path/to/class_names.json
```

## Project Structure

```
mvpdr/
├── train.py              # Training entry point
├── predict.py            # Inference entry point
├── configs/              # Dataset configurations
├── prompts/              # GPT-generated disease descriptions
├── mvpdr/                # Core package
│   ├── clip/             # CLIP model (frozen backbone)
│   ├── datasets/         # Dataset loaders (PlantDoc, PlantVillage, PlantWild)
│   └── utils.py          # Prototypes, metrics, feature extraction
├── scripts/              # Ablation and analysis scripts
└── experiments/          # Saved experiment results
```

## Experiments

Results from ablation studies are saved in `experiments/`:
- **Alpha ablation** — effect of visual-textual fusion weight
- **NCLT ablation** — effect of cluster count per class
- **Backbone comparison** — RN50 vs RN101 vs ViT-B
- **Cross-dataset evaluation** — generalization across benchmarks
- **Reproducibility** — variance across random seeds
- **Open-set detection** — MSP and energy-based OOD scoring

## Citation

If you use this code, please cite:

```bibtex
@misc{gupta2024mvpdr,
  title={MVPDR: Multi-View Prototype-based Disease Recognition},
  author={Gupta, Jatin},
  year={2024}
}
```
