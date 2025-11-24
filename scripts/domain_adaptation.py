import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    '''Gradient Reversal Layer for DANN'''
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainDiscriminator(nn.Module):
    '''Domain classifier for DANN'''
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.layers(x)


class CORAL:
    '''CORrelation ALignment for Domain Adaptation'''
    def __init__(self, lambda_coral=1.0):
        self.lambda_coral = lambda_coral

    def compute_covariance(self, features):
        n = features.size(0)
        features = features - features.mean(dim=0, keepdim=True)
        cov = (features.t() @ features) / (n - 1)
        return cov

    def compute_loss(self, source_features, target_features):
        source_cov = self.compute_covariance(source_features)
        target_cov = self.compute_covariance(target_features)

        d = source_features.size(1)
        loss = torch.norm(source_cov - target_cov, p='fro') ** 2
        loss = loss / (4 * d * d)

        return self.lambda_coral * loss
