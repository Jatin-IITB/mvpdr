# ADR-0008: Phase 2 — Interpretability, Open-Set Detection, Domain Adaptation, Calibration

**Status:** Accepted  
**Date:** 2026-08-07

## Context

Phase 1 established the core architecture upgrades (CoOp, hierarchical prototypes, cross-attention). Phase 2 adds four capabilities that make the system practically deployable and academically rigorous:

1. **Interpretability** — stakeholders (agronomists, farmers) need to understand *why* a model predicts a disease, not just *what* it predicts
2. **Open-set detection** — deployed models encounter unknown diseases and non-disease images; silently misclassifying them is dangerous
3. **Domain adaptation** — lab-quality training images (PlantVillage) differ from field conditions (PlantWild); bridging this gap is critical for real-world use
4. **Calibration** — raw model confidence is typically overconfident; calibrated probabilities are needed for decision-making

## Decisions

### 1. GradCAM for Visual Explanations (`mvpdr/interpretability.py`)

**Approach:** Gradient-weighted Class Activation Mapping using forward hooks on the CLIP visual backbone's last layer.

- **ResNet backbone:** hook into `visual.layer4` → feature maps [B, C, 7, 7]. Weight channels by average gradient → spatial heatmap.
- **ViT backbone:** hook into `visual.transformer.resblocks[-1]` → patch features [L, B, D]. Weight patches by gradient magnitude → reshape to spatial grid.
- Use `retain_grad()` on hooked activations rather than modifying CLIP source code.
- Image input must have `requires_grad=True`; CLIP backbone stays frozen (`requires_grad=False` on params) but gradient flows through for the input image.

**Also includes:** Prototype attention visualization using cross-attention weights from MVPDRPlus — shows which visual/textual prototypes each image attends to. This is a unique contribution of the project.

**Alternatives rejected:**
- Attention rollout: requires modifying CLIP's attention blocks to return weights; violates our "don't modify CLIP" principle.
- SHAP/LIME: model-agnostic but extremely slow for image models (thousands of forward passes per explanation).

### 2. Open-Set Detection (`mvpdr/openset.py`)

**Methods implemented:**
- **MSP** (Maximum Softmax Probability): simplest baseline, `max(softmax(logits))`. Works surprisingly well.
- **Energy score** (Liu et al. 2020): `T * log(Σ exp(f_i/T))`. Theoretically grounded in energy-based models, typically outperforms MSP.
- **Mahalanobis distance** (Lee et al. 2018): fits per-class Gaussian with shared covariance on training features, scores by distance to nearest class centroid. Strongest method but requires fitting step.

**Evaluation metrics:** AUROC, AUPR (in-distribution), FPR@95%TPR. All standard in open-set detection literature.

**Design choice:** scoring functions are stateless (static methods), but Mahalanobis requires a fit step. `fit_mahalanobis()` returns class means and shared precision matrix; `mahalanobis_score()` uses them. This separation keeps the module testable.

### 3. Domain Adaptation (`mvpdr/adaptation.py`)

**Modules:**
- **CORAL** (Sun & Saenko 2016): aligns second-order statistics (covariance) between source and target feature distributions. Loss = ||C_s - C_t||²_F / (4d²). Lightweight, no extra networks.
- **GradientReversal** (Ganin et al. 2016): autograd Function that passes forward unchanged but negates gradients on backward. Enables adversarial training.
- **DomainDiscriminator**: MLP classifier (D→D/2→D/4→1) with gradient reversal. Predicts source vs. target domain; adversarial loss forces the feature extractor to produce domain-invariant features.

**Use case:** PlantVillage (lab) → PlantDoc/PlantWild (field). Source domain has abundant labeled data; target domain has few or no labels.

### 4. Temperature Scaling (`mvpdr/calibration.py`)

**Approach:** post-hoc calibration via a single learnable temperature T, optimized to minimize NLL on a validation set using LBFGS.

**Metrics:**
- **ECE** (Expected Calibration Error): bins predictions by confidence, compares average confidence to accuracy per bin.
- **Reliability diagram**: visual representation of calibration.

**Design choice:** TemperatureScaling is an `nn.Module` with a single parameter. `fit()` runs LBFGS internally. `calibrate()` simply divides logits by T. This post-hoc approach requires no retraining.

## Consequences

- **Positive:** Comprehensive set of tools for deployment readiness (explain, reject unknowns, transfer, calibrate)
- **Positive:** All modules are independent — can be used with baseline MVPDR or MVPDRPlus
- **Positive:** No modification to CLIP source code — everything uses hooks and composition
- **Negative:** GradCAM requires a forward+backward pass per image (not batched efficiently)
- **Negative:** Mahalanobis distance requires computing and inverting a D×D covariance matrix (512×512 = 262K entries) — manageable but not trivial
- **Negative:** Domain adaptation with DANN adds a second training objective, increasing tuning complexity
