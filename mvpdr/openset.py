"""Open-set detection scoring and evaluation.

Implements MSP, Energy, and Mahalanobis distance scoring with standard
evaluation metrics (AUROC, AUPR, FPR@95%TPR).
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


# -----------------------------------------------------------------------
# Scoring functions
# -----------------------------------------------------------------------

def msp_score(logits, temperature=1.0):
    """Maximum Softmax Probability — higher means more in-distribution.

    Args:
        logits:      [B, C] raw logits.
        temperature: softmax temperature (default 1.0).

    Returns:
        scores: [B] confidence scores.
    """
    probs = F.softmax(logits / temperature, dim=-1)
    return probs.max(dim=-1)[0]


def energy_score(logits, temperature=1.0):
    """Energy score (Liu et al. 2020) — higher means more in-distribution.

    Args:
        logits:      [B, C] raw logits.
        temperature: scaling temperature.

    Returns:
        scores: [B] energy values.
    """
    return temperature * torch.logsumexp(logits / temperature, dim=-1)


def fit_mahalanobis(features, labels, n_classes, reg=1e-5):
    """Fit per-class Gaussians with shared covariance for Mahalanobis scoring.

    Args:
        features:  [N, D] L2-normalized features.
        labels:    [N] integer class labels.
        n_classes: number of known classes.
        reg:       Tikhonov regularization for covariance inversion.

    Returns:
        class_means: [C, D] per-class centroids.
        precision:   [D, D] shared precision (inverse covariance) matrix.
    """
    features = features.float()
    class_means = []
    all_centered = []

    for c in range(n_classes):
        mask = labels == c
        cls_feats = features[mask]
        if cls_feats.shape[0] == 0:
            class_means.append(torch.zeros(features.shape[1], device=features.device))
            continue
        mean = cls_feats.mean(dim=0)
        class_means.append(mean)
        all_centered.append(cls_feats - mean)

    class_means = torch.stack(class_means)
    if len(all_centered) == 0:
        d = features.shape[1]
        return class_means, torch.eye(d, device=features.device)

    all_centered = torch.cat(all_centered)
    n = all_centered.shape[0]
    cov = (all_centered.t() @ all_centered) / max(n - 1, 1)
    cov += reg * torch.eye(cov.shape[0], device=cov.device)
    precision = torch.linalg.inv(cov)

    return class_means, precision


def mahalanobis_score(features, class_means, precision):
    """Mahalanobis distance to nearest class — higher means more in-distribution.

    Args:
        features:    [B, D] test features.
        class_means: [C, D] from ``fit_mahalanobis``.
        precision:   [D, D] from ``fit_mahalanobis``.

    Returns:
        scores: [B] negative minimum Mahalanobis distance.
    """
    features = features.float()
    dists = []
    for c in range(class_means.shape[0]):
        diff = features - class_means[c]
        m_dist = (diff @ precision * diff).sum(dim=-1)
        dists.append(m_dist)
    dists = torch.stack(dists, dim=1)
    return -dists.min(dim=1)[0]


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

def evaluate_openset(in_scores, out_scores):
    """Compute standard open-set detection metrics.

    Expects higher scores for in-distribution samples.

    Args:
        in_scores:  [N_in] confidence scores for known-class samples.
        out_scores: [N_out] confidence scores for unknown-class samples.

    Returns:
        dict with auroc, aupr, fpr_at_95tpr, optimal_threshold,
        detection_accuracy, fpr, tpr, thresholds.
    """
    in_np = in_scores.detach().cpu().numpy() if torch.is_tensor(in_scores) else np.asarray(in_scores)
    out_np = out_scores.detach().cpu().numpy() if torch.is_tensor(out_scores) else np.asarray(out_scores)

    labels = np.concatenate([np.ones(len(in_np)), np.zeros(len(out_np))])
    scores = np.concatenate([in_np, out_np])

    auroc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)

    idx_95 = np.searchsorted(tpr, 0.95)
    fpr_at_95tpr = fpr[min(idx_95, len(fpr) - 1)]

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    predictions = (scores >= optimal_threshold).astype(int)
    detection_acc = (predictions == labels).mean()

    prec, rec, _ = precision_recall_curve(labels, scores)
    aupr = auc(rec, prec)

    return {
        "auroc": float(auroc),
        "aupr": float(aupr),
        "fpr_at_95tpr": float(fpr_at_95tpr),
        "optimal_threshold": float(optimal_threshold),
        "detection_accuracy": float(detection_acc),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def save_openset_results(results, output_dir, method_name="openset"):
    """Persist evaluation results (metrics JSON + ROC arrays)."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, f"{method_name}_fpr.npy"), results["fpr"])
    np.save(os.path.join(output_dir, f"{method_name}_tpr.npy"), results["tpr"])

    metrics = {k: v for k, v in results.items() if k not in ("fpr", "tpr", "thresholds")}
    metrics["method"] = method_name
    with open(os.path.join(output_dir, f"{method_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def plot_openset_roc(results_dict, save_path=None):
    """Plot ROC curves for one or more open-set methods.

    Args:
        results_dict: {method_name: evaluate_openset() output}.
        save_path:    optional file path.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results_dict.items():
        ax.plot(res["fpr"], res["tpr"],
                label=f"{name} (AUROC={res['auroc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Open-Set Detection ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
