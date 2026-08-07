import torch
import torch.nn as nn
import torch.nn.functional as F

from mvpdr.models.cross_attention import CrossAttentionFusion
from mvpdr.models.prompt_learner import PromptLearner
from mvpdr.models.prototype_bank import HierarchicalPrototypeBank


class MVPDRPlus(nn.Module):
    """Full MVPDR+ model with configurable architecture components.

    Three independently toggleable upgrades over the baseline:

    1. **PromptLearner** — CoOp-style learnable soft prompts replace the
       fixed CLIP text encoder, letting the model discover optimal text
       context for each dataset.

    2. **HierarchicalPrototypeBank** — multi-granularity visual prototypes
       (e.g. 4/8/16 clusters per class) with a learned router that picks
       the right resolution per sample.

    3. **CrossAttentionFusion** — image features attend over all prototypes
       (visual + textual) through stacked cross-attention layers, replacing
       the baseline's fixed weighted sum.

    Disable any component via the config dict for ablation studies.
    """

    def __init__(self, clip_model, classnames, config):
        super().__init__()

        embed_dim = clip_model.text_projection.shape[1]
        n_classes = len(classnames)

        # --- Prompt Learner ---
        self.use_prompt_learner = config.get("use_prompt_learner", True)
        if self.use_prompt_learner:
            self.prompt_learner = PromptLearner(
                clip_model,
                classnames,
                n_ctx=config.get("n_ctx", 4),
                ctx_init=config.get("ctx_init", ""),
                class_specific=config.get("class_specific_ctx", False),
            )
        else:
            self.prompt_learner = None

        # --- Hierarchical Prototype Bank ---
        self.use_prototype_bank = config.get("use_prototype_bank", True)
        if self.use_prototype_bank:
            self.prototype_bank = HierarchicalPrototypeBank(
                embed_dim,
                n_classes,
                levels=config.get("prototype_levels", (4, 8, 16)),
                momentum=config.get("proto_momentum", 0.999),
            )
        else:
            self.prototype_bank = None

        # --- Cross-Attention Fusion ---
        self.use_cross_attn = config.get("use_cross_attn", True)
        if self.use_cross_attn:
            self.cross_attn = CrossAttentionFusion(
                embed_dim,
                n_heads=config.get("n_heads", 8),
                n_layers=config.get("n_cross_layers", 2),
                dropout=config.get("dropout", 0.1),
            )
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        else:
            self.cross_attn = None
            self.alpha = config.get("alpha", 0.3)

        self.n_classes = n_classes
        self.embed_dim = embed_dim

    # ------------------------------------------------------------------

    def forward(self, image_features, clip_model, text_features=None):
        """
        Args:
            image_features: [B, D] L2-normalized CLIP image features
            clip_model:     frozen CLIP model (used by prompt learner)
            text_features:  [C, D] pre-computed text features (optional —
                            required when prompt_learner is disabled)

        Returns:
            logits: [B, C] classification logits
            aux:    dict with ``textual_logits``, ``visual_logits``,
                    ``attn_weights`` (present only when the corresponding
                    component is enabled)
        """
        aux = {}

        # ---- text features ----
        if self.prompt_learner is not None:
            text_features = self.prompt_learner(clip_model)
            text_features = F.normalize(text_features, dim=-1)
        elif text_features is not None:
            text_features = F.normalize(text_features.float(), dim=-1)
        else:
            raise ValueError(
                "No text features: enable prompt_learner or pass text_features"
            )

        t_logits = image_features.float() @ text_features.float().t()
        aux["textual_logits"] = t_logits

        # ---- visual logits ----
        if self.prototype_bank is not None:
            v_logits = self.prototype_bank(image_features)
            aux["visual_logits"] = v_logits

        # ---- fusion ----
        if self.cross_attn is not None:
            proto_list = [text_features.float()]
            if self.prototype_bank is not None:
                all_protos = self.prototype_bank.get_all_prototypes()
                proto_list.append(F.normalize(all_protos.float(), dim=-1))

            all_prototypes = torch.cat(proto_list, dim=0)
            fused, attn_weights = self.cross_attn(image_features, all_prototypes)
            fused = F.normalize(fused, dim=-1)

            logits = self.logit_scale.exp() * fused @ text_features.float().t()
            aux["attn_weights"] = attn_weights
        else:
            logits = t_logits
            if self.prototype_bank is not None:
                logits = t_logits + self.alpha * v_logits

        return logits, aux
