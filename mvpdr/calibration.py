"""Post-hoc calibration via temperature scaling.

Learns a single scalar temperature T on a validation set that minimizes
NLL, then applies ``logits / T`` at test time to produce calibrated
probabilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaling(nn.Module):
    """Learnable temperature for post-hoc logit calibration."""

    def __init__(self, init_temp=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(init_temp)))

    @torch.enable_grad()
    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """Find optimal T via LBFGS on a held-out validation set.

        Args:
            logits:   [N, C] raw (uncalibrated) logits.
            labels:   [N] integer class labels.
            lr:       LBFGS learning rate.
            max_iter: maximum LBFGS iterations.

        Returns:
            Learned temperature value (float).
        """
        logits = logits.detach().float()
        labels = labels.detach().long()

        self.temperature.data.fill_(1.5)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.temperature.data.clamp_(min=0.01)
        return self.temperature.item()

    def calibrate(self, logits):
        """Apply learned temperature to logits.

        Args:
            logits: [B, C] raw logits.

        Returns:
            Scaled logits [B, C].
        """
        return logits / self.temperature

    def calibrated_probs(self, logits):
        """Return calibrated softmax probabilities."""
        return F.softmax(self.calibrate(logits), dim=-1)


# -----------------------------------------------------------------------
# Calibration metrics
# -----------------------------------------------------------------------

def expected_calibration_error(probs, labels, n_bins=15):
    """Expected Calibration Error (ECE).

    Args:
        probs:  [N, C] softmax probabilities.
        labels: [N] true class labels.
        n_bins: number of confidence bins.

    Returns:
        ECE value (float in [0, 1]).
    """
    if torch.is_tensor(probs):
        confs, preds = probs.max(dim=-1)
        confs = confs.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
    else:
        confs = np.max(probs, axis=-1)
        preds = np.argmax(probs, axis=-1)

    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels_np)

    for i in range(n_bins):
        mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = confs[mask].mean()
        avg_acc = (preds[mask] == labels_np[mask]).mean()
        ece += (count / n) * abs(avg_conf - avg_acc)

    return float(ece)


def reliability_diagram(probs, labels, n_bins=15, save_path=None):
    """Plot a reliability diagram showing calibration.

    Args:
        probs:     [N, C] softmax probabilities.
        labels:    [N] true class labels.
        n_bins:    number of confidence bins.
        save_path: optional file path to save the plot.
    """
    import matplotlib.pyplot as plt

    if torch.is_tensor(probs):
        confs, preds = probs.max(dim=-1)
        confs = confs.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
    else:
        confs = np.max(probs, axis=-1)
        preds = np.argmax(probs, axis=-1)

    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_mids = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        bin_accs[i] = (preds[mask] == labels_np[mask]).mean()
        bin_confs[i] = confs[mask].mean()
        bin_counts[i] = count

    ece = expected_calibration_error(probs, labels, n_bins)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Reliability plot
    ax1.bar(bin_mids, bin_accs, width=1 / n_bins, alpha=0.6,
            edgecolor="black", label="Accuracy")
    ax1.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Reliability Diagram (ECE = {ece:.4f})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Confidence histogram
    ax2.bar(bin_mids, bin_counts / max(bin_counts.sum(), 1),
            width=1 / n_bins, alpha=0.6, edgecolor="black", color="orange")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Fraction of samples")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
