import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """Pre-norm cross-attention block: LayerNorm → MHA → residual → LayerNorm → FFN → residual."""

    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, key_value):
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attn_out, attn_weights = self.attn(q, kv, kv)
        query = query + attn_out
        query = query + self.ffn(self.norm_ffn(query))
        return query, attn_weights


class CrossAttentionFusion(nn.Module):
    """Stack of cross-attention layers for prototype-guided image feature refinement.

    Image features (query) attend over concatenated visual + textual prototypes
    (key/value) to produce a fused representation that captures the most
    relevant information from both views.
    """

    def __init__(self, embed_dim, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, image_features, prototypes):
        """Cross-attend image features over prototype bank.

        Args:
            image_features: [B, D] — CLIP image features
            prototypes:     [N, D] — concatenated visual + textual prototypes

        Returns:
            fused:        [B, D] — refined image features
            attn_weights: [B, N] — last-layer attention weights (for visualization)
        """
        query = image_features.unsqueeze(1).float()        # [B, 1, D]
        kv = prototypes.unsqueeze(0).expand(query.shape[0], -1, -1).float()  # [B, N, D]

        attn_weights = None
        for layer in self.layers:
            query, attn_weights = layer(query, kv)

        fused = self.norm_out(query).squeeze(1)   # [B, D]
        attn_weights = attn_weights.squeeze(1)    # [B, N]

        return fused, attn_weights
