"""Domain adaptation modules for cross-domain plant disease recognition.

Provides CORAL alignment and DANN (gradient reversal + domain discriminator)
for bridging the gap between lab imagery (PlantVillage) and field conditions
(PlantDoc / PlantWild).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# CORAL — CORrelation ALignment (Sun & Saenko 2016)
# -----------------------------------------------------------------------

class CORAL(nn.Module):
    """Minimize the Frobenius-norm distance between source and target
    feature covariance matrices: ||C_s - C_t||²_F / (4d²).
    """

    def __init__(self, lambda_coral=1.0):
        super().__init__()
        self.lambda_coral = lambda_coral

    @staticmethod
    def _covariance(features):
        n = features.size(0)
        centered = features - features.mean(dim=0, keepdim=True)
        return (centered.t() @ centered) / max(n - 1, 1)

    def forward(self, source_features, target_features):
        """Compute CORAL loss between source and target feature batches.

        Args:
            source_features: [B_s, D] source-domain features.
            target_features: [B_t, D] target-domain features.

        Returns:
            Scalar CORAL loss.
        """
        src = source_features.float()
        tgt = target_features.float()
        cov_s = self._covariance(src)
        cov_t = self._covariance(tgt)
        d = src.size(1)
        loss = torch.norm(cov_s - cov_t, p="fro") ** 2 / (4 * d * d)
        return self.lambda_coral * loss


# -----------------------------------------------------------------------
# DANN — Domain-Adversarial Neural Networks (Ganin et al. 2016)
# -----------------------------------------------------------------------

class _GradientReversal(torch.autograd.Function):
    """Passes input forward unchanged; negates gradients on backward."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x, alpha=1.0):
    """Functional interface for gradient reversal."""
    return _GradientReversal.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    """MLP domain classifier with built-in gradient reversal.

    Architecture: D → D/2 → D/4 → 1 (binary cross-entropy).
    Gradient reversal is applied at the input so the upstream feature
    extractor is pushed toward domain-invariant representations.
    """

    def __init__(self, embed_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(self, features, alpha=1.0):
        """Predict domain label with gradient reversal.

        Args:
            features: [B, D] image features.
            alpha:    gradient reversal strength (ramp up during training).

        Returns:
            domain_logits: [B, 1] — 1 = source, 0 = target.
        """
        x = gradient_reversal(features.float(), alpha)
        return self.net(x)


def dann_loss(domain_logits, domain_labels):
    """Binary cross-entropy loss for the domain discriminator.

    Args:
        domain_logits: [B, 1] from DomainDiscriminator.
        domain_labels: [B] float tensor (1.0 = source, 0.0 = target).

    Returns:
        Scalar loss.
    """
    return F.binary_cross_entropy_with_logits(
        domain_logits.squeeze(-1), domain_labels,
    )


def grl_alpha_schedule(epoch, total_epochs):
    """Gradient reversal strength schedule (Ganin et al.):
    α = 2 / (1 + exp(-10 * p)) - 1  where p = epoch / total_epochs.
    Ramps from 0 → 1 over training.
    """
    import math
    p = epoch / max(total_epochs, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
