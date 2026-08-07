import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class HierarchicalPrototypeBank(nn.Module):
    """Multi-granularity visual prototype bank with learned routing.

    Maintains K-Means prototypes at several granularity levels (e.g. 4, 8, 16
    clusters per class).  A lightweight router network learns per-sample
    weights over levels so that easy examples rely on coarse prototypes while
    hard examples benefit from fine-grained ones.

    Prototypes are learnable parameters initialized from K-Means and
    optionally refined with momentum updates during training.
    """

    def __init__(self, embed_dim, n_classes, levels=(4, 8, 16), momentum=0.999):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.levels = list(levels)
        self.n_levels = len(levels)
        self.momentum = momentum

        self.prototypes = nn.ParameterList()
        for k in levels:
            protos = torch.empty(n_classes * k, embed_dim)
            nn.init.normal_(protos, std=0.02)
            self.prototypes.append(nn.Parameter(protos))

        for level_idx, k in enumerate(levels):
            labels = torch.arange(n_classes).repeat_interleave(k)
            one_hot = F.one_hot(labels, n_classes).float()
            self.register_buffer(f"labels_{level_idx}", one_hot)

        self.router = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, self.n_levels),
        )

        self.beta = nn.Parameter(torch.full((self.n_levels,), 0.5))

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_from_features(self, features, labels):
        """Initialize each level's prototypes via K-Means on training features."""
        features_np = features.detach().cpu().float().numpy()
        labels_np = labels.detach().cpu().long().numpy()

        for level_idx, k in enumerate(self.levels):
            centroids = []
            for cls in range(self.n_classes):
                cls_feats = features_np[labels_np == cls]
                n_samples = cls_feats.shape[0]
                effective_k = min(k, max(1, n_samples))

                if n_samples == 0:
                    centroid = np.random.randn(k, self.embed_dim).astype(np.float32)
                    centroid /= np.linalg.norm(centroid, axis=-1, keepdims=True) + 1e-8
                elif n_samples == 1:
                    centroid = cls_feats
                elif effective_k == 1:
                    centroid = cls_feats.mean(axis=0, keepdims=True)
                else:
                    kmeans = KMeans(n_clusters=effective_k, random_state=0, n_init=10)
                    kmeans.fit(cls_feats)
                    centroid = kmeans.cluster_centers_

                while centroid.shape[0] < k:
                    noise = np.random.randn(1, self.embed_dim).astype(np.float32) * 0.01
                    centroid = np.concatenate([centroid, centroid[:1] + noise], axis=0)
                centroids.append(centroid[:k])

            centroids = np.concatenate(centroids, axis=0)
            centroids_t = torch.from_numpy(centroids).to(self.prototypes[level_idx].dtype)
            centroids_t = F.normalize(centroids_t, dim=-1)
            self.prototypes[level_idx].data.copy_(centroids_t)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, image_features):
        """Compute router-weighted prototype logits.

        Args:
            image_features: [B, D] L2-normalized CLIP image features

        Returns:
            logits: [B, C] classification logits
        """
        x = image_features.float()
        route_weights = F.softmax(self.router(x), dim=-1)

        logits = torch.zeros(x.shape[0], self.n_classes, device=x.device)
        for level_idx in range(self.n_levels):
            protos = F.normalize(self.prototypes[level_idx].float(), dim=-1)
            one_hot = getattr(self, f"labels_{level_idx}").float()
            beta = F.softplus(self.beta[level_idx])

            affinity = x @ protos.t()
            level_logits = ((-1) * (beta - beta * affinity)).exp() @ one_hot
            logits = logits + route_weights[:, level_idx:level_idx + 1] * level_logits

        return logits

    def get_all_prototypes(self):
        """Return all prototypes concatenated across levels.

        Returns:
            prototypes: [sum(C*k_i), D]
        """
        return torch.cat(list(self.prototypes), dim=0)

    # ------------------------------------------------------------------
    # Momentum update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def momentum_update(self, features, labels):
        """EMA update: move each prototype toward its matched training features."""
        features = features.float()

        for level_idx in range(self.n_levels):
            protos = self.prototypes[level_idx].data.float()
            proto_cls = getattr(self, f"labels_{level_idx}").argmax(dim=-1)

            for cls in labels.unique():
                cls_val = cls.item()
                cls_feats = features[labels == cls_val]
                proto_indices = (proto_cls == cls_val).nonzero(as_tuple=True)[0]
                cls_protos = protos[proto_indices]

                if len(cls_feats) == 0 or len(cls_protos) == 0:
                    continue

                sims = F.normalize(cls_feats, dim=-1) @ F.normalize(cls_protos, dim=-1).t()
                assignments = sims.argmax(dim=1)

                for p_local in range(cls_protos.shape[0]):
                    matched = cls_feats[assignments == p_local]
                    if len(matched) == 0:
                        continue
                    mean_feat = F.normalize(matched.mean(0), dim=0)
                    p_global = proto_indices[p_local]
                    protos[p_global] = (
                        self.momentum * protos[p_global]
                        + (1 - self.momentum) * mean_feat
                    )
                    protos[p_global] = F.normalize(protos[p_global], dim=0)

            self.prototypes[level_idx].data.copy_(
                protos.to(self.prototypes[level_idx].dtype)
            )
