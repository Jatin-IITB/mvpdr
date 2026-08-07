import torch
import torch.nn as nn

from mvpdr import clip


class PromptLearner(nn.Module):
    """CoOp-style learnable soft prompts for CLIP text encoder.

    Replaces fixed text templates with learnable continuous context vectors
    optimized end-to-end. Per-class prompt structure:

        [SOS] [v_1] ... [v_m] [class_name_tokens] [EOS] [PAD...]

    where v_1..v_m are m learnable vectors shared across classes (or
    per-class if class_specific=True).
    """

    def __init__(self, clip_model, classnames, n_ctx=4, ctx_init="", class_specific=False):
        super().__init__()

        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        ctx_length = clip_model.context_length

        classnames = [name.replace("_", " ") for name in classnames]

        # Tokenize each class name to find its length
        name_lens = []
        for name in classnames:
            tokens = clip.tokenize([name])[0]
            name_lens.append(tokens.argmax().item() - 1)

        max_name_len = max(name_lens)
        assert 2 + n_ctx + max_name_len <= ctx_length, (
            f"Prompt too long: SOS + {n_ctx}ctx + {max_name_len}name + EOS "
            f"= {2 + n_ctx + max_name_len} > {ctx_length}"
        )

        # Extract token embeddings for class names
        with torch.no_grad():
            all_tokens = clip.tokenize(classnames)
            all_emb = clip_model.token_embedding(all_tokens).type(dtype)

            sot_emb = all_emb[0, 0:1]

            suffix_len = max_name_len + 1  # class tokens + EOS
            suffix = torch.zeros(n_cls, suffix_len, ctx_dim, dtype=dtype)
            eos_positions = []

            for i in range(n_cls):
                eos_pos = all_tokens[i].argmax().item()
                name_len = eos_pos - 1
                suffix[i, :name_len] = all_emb[i, 1:eos_pos]
                suffix[i, name_len] = all_emb[i, eos_pos]
                eos_positions.append(1 + n_ctx + name_len)

        self.register_buffer("sot_emb", sot_emb)
        self.register_buffer("suffix", suffix)
        self.register_buffer("eos_positions", torch.tensor(eos_positions))

        # Learnable context vectors — optionally initialized from a text string
        if ctx_init:
            init_tokens = clip.tokenize([ctx_init.replace("_", " ")])[0]
            with torch.no_grad():
                init_emb = clip_model.token_embedding(init_tokens.unsqueeze(0)).type(dtype)[0]
            n_use = min(n_ctx, init_tokens.argmax().item() - 1)
            ctx_vectors = init_emb[1:1 + n_use].clone()
            if n_use < n_ctx:
                pad = torch.empty(n_ctx - n_use, ctx_dim, dtype=dtype)
                nn.init.normal_(pad, std=0.02)
                ctx_vectors = torch.cat([ctx_vectors, pad])
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        if class_specific:
            ctx_vectors = ctx_vectors.unsqueeze(0).expand(n_cls, -1, -1).clone()

        self.ctx = nn.Parameter(ctx_vectors)
        self.n_ctx = n_ctx
        self.n_cls = n_cls
        self.class_specific = class_specific
        self.ctx_length = ctx_length
        self.ctx_dim = ctx_dim

    def forward(self, clip_model):
        """Compute text features for all classes using learnable prompts.

        Args:
            clip_model: frozen CLIP model (provides transformer, ln_final,
                        positional_embedding, text_projection)

        Returns:
            text_features: [n_cls, embed_dim] — not L2-normalized
        """
        ctx = self.ctx
        dtype = ctx.dtype
        device = ctx.device

        if not self.class_specific:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = torch.zeros(
            self.n_cls, self.ctx_length, self.ctx_dim, dtype=dtype, device=device,
        )
        prompts[:, 0] = self.sot_emb[0]
        prompts[:, 1:1 + self.n_ctx] = ctx
        start = 1 + self.n_ctx
        prompts[:, start:start + self.suffix.shape[1]] = self.suffix

        prompts = prompts + clip_model.positional_embedding.type(dtype)

        x = prompts.permute(1, 0, 2)  # NLD → LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND → NLD
        x = clip_model.ln_final(x).type(dtype)

        x = x[torch.arange(self.n_cls, device=device), self.eos_positions]
        x = x @ clip_model.text_projection.type(dtype)

        return x
