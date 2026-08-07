"""Severity estimation for plant diseases.

Two approaches:
1. **Heuristic** — label-free severity from GradCAM coverage, classification
   confidence, and feature distance from healthy prototypes.  Works without
   any severity annotations.
2. **Ordinal regression head** — trainable CORAL ordinal head for when
   labeled severity data is available.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeverityLevel(IntEnum):
    HEALTHY = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3


SEVERITY_LABELS = {
    SeverityLevel.HEALTHY: "Healthy",
    SeverityLevel.MILD: "Mild",
    SeverityLevel.MODERATE: "Moderate",
    SeverityLevel.SEVERE: "Severe",
}

SEVERITY_COLORS = {
    SeverityLevel.HEALTHY: "#27ae60",
    SeverityLevel.MILD: "#f39c12",
    SeverityLevel.MODERATE: "#e67e22",
    SeverityLevel.SEVERE: "#e74c3c",
}


@dataclass
class SeverityResult:
    level: SeverityLevel
    label: str
    score: float
    gradcam_coverage: float
    confidence: float
    feature_distance: Optional[float] = None
    evidence: dict = field(default_factory=dict)


# -----------------------------------------------------------------------
# Heuristic severity estimation (no labels required)
# -----------------------------------------------------------------------

def estimate_severity_heuristic(
    gradcam_map: np.ndarray,
    classification_confidence: float,
    feature_distance: Optional[float] = None,
    cam_threshold: float = 0.3,
    weights: tuple = (0.4, 0.3, 0.3),
) -> SeverityResult:
    """Estimate disease severity without labeled severity data.

    Combines three proxy signals:

    1. **GradCAM coverage** — fraction of spatial locations above
       ``cam_threshold``.  More affected area → higher severity.
    2. **Classification confidence** — high confidence in a disease class
       suggests clear, advanced symptoms.
    3. **Feature distance** — distance of the image embedding from the
       healthy-class centroid (optional; skipped if None).

    Args:
        gradcam_map:                [H, W] heatmap in [0, 1].
        classification_confidence:  top-1 softmax probability (0-1).
        feature_distance:           cosine distance from healthy centroid.
        cam_threshold:              activation threshold for coverage.
        weights:                    (w_coverage, w_confidence, w_distance).

    Returns:
        SeverityResult with level, score, and per-signal evidence.
    """
    coverage = float((gradcam_map > cam_threshold).mean())
    w_cov, w_conf, w_dist = weights

    score = w_cov * coverage + w_conf * classification_confidence

    if feature_distance is not None:
        dist_norm = min(feature_distance / 2.0, 1.0)
        score += w_dist * dist_norm
    else:
        score = score / (w_cov + w_conf) if (w_cov + w_conf) > 0 else score

    score = float(np.clip(score, 0.0, 1.0))

    if score < 0.2:
        level = SeverityLevel.HEALTHY
    elif score < 0.45:
        level = SeverityLevel.MILD
    elif score < 0.7:
        level = SeverityLevel.MODERATE
    else:
        level = SeverityLevel.SEVERE

    return SeverityResult(
        level=level,
        label=SEVERITY_LABELS[level],
        score=score,
        gradcam_coverage=coverage,
        confidence=classification_confidence,
        feature_distance=feature_distance,
        evidence={
            "cam_coverage_pct": round(coverage * 100, 1),
            "cam_threshold": cam_threshold,
            "classification_confidence_pct": round(classification_confidence * 100, 1),
            "feature_distance_raw": feature_distance,
            "weighted_score": round(score, 4),
        },
    )


def compute_healthy_distance(
    image_features: torch.Tensor,
    all_features: torch.Tensor,
    all_labels: torch.Tensor,
    healthy_class_idx: int,
) -> float:
    """Compute cosine distance from the healthy-class centroid.

    Args:
        image_features:   [1, D] or [D] query features.
        all_features:     [N, D] training set features.
        all_labels:       [N] class labels.
        healthy_class_idx: integer label for the healthy class.

    Returns:
        Cosine distance (1 - cosine_similarity) to the healthy centroid.
    """
    mask = all_labels == healthy_class_idx
    if mask.sum() == 0:
        return 1.0

    healthy_feats = all_features[mask].float()
    centroid = F.normalize(healthy_feats.mean(dim=0, keepdim=True), dim=-1)
    query = F.normalize(image_features.float().reshape(1, -1), dim=-1)
    sim = (query @ centroid.t()).item()
    return float(1.0 - sim)


# -----------------------------------------------------------------------
# Trainable ordinal severity head (CORAL)
# -----------------------------------------------------------------------

class OrdinalSeverityHead(nn.Module):
    """CORAL ordinal regression for severity grading.

    When labeled severity data (0=healthy, 1=mild, 2=moderate, 3=severe)
    is available, this head learns ordinal thresholds on top of CLIP
    features.

    Reference: Cao et al., "Rank consistent ordinal regression for neural
    networks with application to age estimation", Pattern Recognition 2020.
    """

    def __init__(self, embed_dim: int, n_levels: int = 4):
        super().__init__()
        self.n_levels = n_levels
        self.fc = nn.Linear(embed_dim, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(n_levels - 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict ordinal cumulative logits.

        Args:
            features: [B, D] image features.

        Returns:
            cumulative_logits: [B, n_levels-1] logits for P(Y > k).
        """
        logit = self.fc(features)
        return logit + self.biases

    def predict_level(self, features: torch.Tensor) -> torch.Tensor:
        """Predict severity level (0 to n_levels-1).

        Args:
            features: [B, D] image features.

        Returns:
            levels: [B] predicted severity level.
        """
        cum_logits = self.forward(features)
        cum_probs = torch.sigmoid(cum_logits)
        return (cum_probs > 0.5).sum(dim=-1).long()

    def predict_probs(self, features: torch.Tensor) -> torch.Tensor:
        """Predict per-level probabilities.

        Args:
            features: [B, D] image features.

        Returns:
            probs: [B, n_levels] probability of each severity level.
        """
        cum_logits = self.forward(features)
        cum_probs = torch.sigmoid(cum_logits)

        probs = torch.zeros(
            features.shape[0], self.n_levels, device=features.device
        )
        probs[:, 0] = 1.0 - cum_probs[:, 0]
        for k in range(1, self.n_levels - 1):
            probs[:, k] = cum_probs[:, k - 1] - cum_probs[:, k]
        probs[:, -1] = cum_probs[:, -1]
        return probs.clamp(min=0.0)


def coral_loss(
    cumulative_logits: torch.Tensor,
    levels: torch.Tensor,
    n_levels: int = 4,
) -> torch.Tensor:
    """CORAL ordinal regression loss.

    Args:
        cumulative_logits: [B, n_levels-1] from OrdinalSeverityHead.
        levels:            [B] integer severity levels (0 to n_levels-1).
        n_levels:          total number of ordinal levels.

    Returns:
        Scalar loss.
    """
    targets = torch.zeros_like(cumulative_logits)
    for k in range(n_levels - 1):
        targets[:, k] = (levels > k).float()
    return F.binary_cross_entropy_with_logits(cumulative_logits, targets)


# -----------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------

def plot_severity(severity_result: SeverityResult, save_path=None):
    """Render a severity gauge visualization."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3),
                                    gridspec_kw={"width_ratios": [3, 2]})

    levels = list(SEVERITY_LABELS.values())
    colors = list(SEVERITY_COLORS.values())
    positions = np.linspace(0, 1, len(levels))
    width = 1.0 / len(levels)

    for i, (pos, color, label) in enumerate(zip(positions, colors, levels)):
        alpha = 1.0 if i == severity_result.level else 0.3
        ax1.barh(0, width, left=pos, color=color, alpha=alpha,
                 edgecolor="white", height=0.5)
        ax1.text(pos + width / 2, -0.05, label, ha="center", va="top",
                 fontsize=9, fontweight="bold" if i == severity_result.level else "normal")

    ax1.axvline(x=severity_result.score, color="black", linewidth=2.5,
                linestyle="--", zorder=5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.3, 0.4)
    ax1.set_yticks([])
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_xlabel("Severity Score")
    ax1.set_title(f"Severity: {severity_result.label}", fontsize=12,
                  fontweight="bold")

    evidence = [
        f"GradCAM Coverage: {severity_result.evidence['cam_coverage_pct']}%",
        f"Classification Conf: {severity_result.evidence['classification_confidence_pct']}%",
    ]
    if severity_result.feature_distance is not None:
        evidence.append(f"Healthy Distance: {severity_result.feature_distance:.3f}")

    ax2.axis("off")
    ax2.set_title("Evidence", fontsize=11, fontweight="bold")
    for i, line in enumerate(evidence):
        ax2.text(0.1, 0.8 - i * 0.25, line, fontsize=10,
                 transform=ax2.transAxes, va="top")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
