"""Visual interpretability for CLIP-based models.

Provides GradCAM heatmaps for both ResNet and ViT CLIP backbones, plus
prototype attention visualization unique to MVPDRPlus.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mvpdr.clip.model import ModifiedResNet


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CLIP visual backbones.

    Works with both ResNet (hooks into layer4) and ViT (hooks into the last
    transformer block).  The CLIP backbone stays frozen — gradients flow
    through it only because the input image has ``requires_grad=True``.
    """

    def __init__(self, clip_model):
        self.clip_model = clip_model
        self._feature_map = None
        self._hooks = []

        if isinstance(clip_model.visual, ModifiedResNet):
            target = clip_model.visual.layer4
            self._backbone = "resnet"
        else:
            target = clip_model.visual.transformer.resblocks[-1]
            self._backbone = "vit"

        self._hooks.append(target.register_forward_hook(self._capture))

    def _capture(self, module, inp, out):
        self._feature_map = out
        if out.requires_grad or (out.is_leaf is False):
            out.retain_grad()

    @torch.enable_grad()
    def generate(self, image, text_features, class_idx=None):
        """Compute a GradCAM heatmap for one image.

        Args:
            image:         [1, 3, H, W] preprocessed image tensor.
            text_features: [C, D] L2-normalized text features.
            class_idx:     target class (default: argmax prediction).

        Returns:
            cam:       [H', W'] numpy heatmap in [0, 1].
            class_idx: the class used for the gradient signal.
        """
        image = image.clone().detach().requires_grad_(True)
        self._feature_map = None

        image_features = self.clip_model.encode_image(image)
        if image_features.dim() > 2:
            image_features = image_features[:, 0, :]
        image_features = F.normalize(image_features.float(), dim=-1)

        logits = 100.0 * image_features @ text_features.float().t()

        if class_idx is None:
            class_idx = logits[0].argmax().item()

        self.clip_model.zero_grad()
        logits[0, class_idx].backward(retain_graph=False)

        fm = self._feature_map
        if fm is None:
            raise RuntimeError(
                "GradCAM forward hook did not fire — is the target layer "
                "present in the model?"
            )
        grad = fm.grad
        if grad is None:
            raise RuntimeError(
                "GradCAM backward hook did not capture gradients — ensure "
                "no torch.no_grad() context wraps the forward pass."
            )

        if self._backbone == "resnet":
            weights = grad.mean(dim=[2, 3], keepdim=True)
            cam = (weights * fm).sum(dim=1)
        else:
            # ViT: fm is [L, B, D] in LND format — requires batch_size=1
            assert fm.shape[1] == 1, (
                f"ViT GradCAM requires batch_size=1, got {fm.shape[1]}"
            )
            patches = fm[1:]
            g = grad[1:]
            weights = g.mean(dim=-1, keepdim=True)
            cam = (weights * patches).sum(dim=-1)
            cam = cam.permute(1, 0)
            side = int(cam.shape[1] ** 0.5)
            cam = cam.reshape(cam.shape[0], side, side)

        cam = F.relu(cam)
        cam = cam.squeeze(0).detach().cpu().float().numpy()
        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam, class_idx

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def plot_gradcam(image_pil, cam, class_name=None, save_path=None):
    """Render a three-panel GradCAM figure (original | heatmap | overlay).

    Args:
        image_pil:  PIL.Image of the original input.
        cam:        [H, W] numpy heatmap in [0, 1] (from GradCAM.generate).
        class_name: optional class label for the figure title.
        save_path:  if given, save to this path instead of plt.show().
    """
    import matplotlib.pyplot as plt

    cam_resized = np.array(
        Image.fromarray(np.uint8(cam * 255)).resize(
            image_pil.size, resample=Image.BILINEAR,
        )
    ) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_pil)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cam_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("GradCAM")
    axes[1].axis("off")

    axes[2].imshow(image_pil)
    axes[2].imshow(cam_resized, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if class_name:
        fig.suptitle(f"Predicted: {class_name}", fontsize=14)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_prototype_attention(
    attn_weights,
    n_text_protos,
    classnames,
    prototype_levels,
    save_path=None,
    top_k=15,
):
    """Visualize cross-attention weights over textual and visual prototypes.

    Args:
        attn_weights:     [N] attention over concatenated prototypes (one sample).
        n_text_protos:    number of textual prototypes (= n_classes).
        classnames:       list of class names.
        prototype_levels: e.g. [4, 8, 16].
        save_path:        optional save path.
        top_k:            number of top-attended prototypes to show.
    """
    import matplotlib.pyplot as plt

    attn = attn_weights.detach().cpu().float().numpy()
    n_classes = len(classnames)

    text_attn = attn[:n_text_protos]
    visual_attn = attn[n_text_protos:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    top_text = np.argsort(text_attn)[::-1][:top_k]
    axes[0].barh(
        range(len(top_text)),
        text_attn[top_text],
        color="steelblue",
    )
    axes[0].set_yticks(range(len(top_text)))
    axes[0].set_yticklabels([classnames[i] for i in top_text])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Attention Weight")
    axes[0].set_title("Textual Prototype Attention")

    level_labels = []
    level_colors = []
    palette = plt.cm.Set2(np.linspace(0, 1, len(prototype_levels)))
    offset = 0
    for li, k in enumerate(prototype_levels):
        n_level = n_classes * k
        for pi in range(n_level):
            cls = pi // k
            level_labels.append(f"L{li}-C{cls}")
            level_colors.append(palette[li])
        offset += n_level

    if len(visual_attn) > 0:
        top_vis = np.argsort(visual_attn)[::-1][:top_k]
        axes[1].barh(
            range(len(top_vis)),
            visual_attn[top_vis],
            color=[level_colors[i] for i in top_vis],
        )
        axes[1].set_yticks(range(len(top_vis)))
        axes[1].set_yticklabels([level_labels[i] for i in top_vis])
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Attention Weight")
        axes[1].set_title("Visual Prototype Attention")

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color=palette[i], label=f"Level {i} (k={k})")
            for i, k in enumerate(prototype_levels)
        ]
        axes[1].legend(handles=legend_handles, loc="lower right")
    else:
        axes[1].text(0.5, 0.5, "No visual prototypes", ha="center", va="center",
                     transform=axes[1].transAxes)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
