"""Generate GradCAM visualizations for MVPDR+ predictions.

Usage:
    python scripts/visualize_gradcam.py \\
        --image path/to/image.jpg \\
        --config configs/plantdoc_plus.yaml \\
        --checkpoint results/.../best_model.pth \\
        --output gradcam_output/
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mvpdr import clip
from mvpdr.interpretability import GradCAM, plot_gradcam, plot_prototype_attention
from mvpdr.models import MVPDRPlus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="gradcam_output")
    parser.add_argument("--class_idx", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    clip_model, preprocess = clip.load(cfg["backbone"])
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
    classnames = ckpt["classnames"] if "classnames" in ckpt else None

    if classnames is None:
        from mvpdr.datasets import build_dataset
        dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])
        classnames = dataset.classnames

    cfg["n_classes"] = len(classnames)
    model = MVPDRPlus(clip_model, classnames, cfg).to(device)

    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # ---- load and preprocess image ----
    image_pil = Image.open(args.image).convert("RGB")
    image_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    # ---- get text features for GradCAM target ----
    with torch.no_grad():
        if model.prompt_learner is not None:
            text_features = model.prompt_learner(clip_model)
            text_features = F.normalize(text_features, dim=-1)
        else:
            from mvpdr.utils import clip_classifier
            template = ["a photo of a {}."]
            text_features = clip_classifier(classnames, template, clip_model).t()

    # ---- GradCAM ----
    gradcam = GradCAM(clip_model)
    cam, pred_idx = gradcam.generate(image_tensor, text_features, class_idx=args.class_idx)
    gradcam.remove_hooks()

    pred_name = classnames[pred_idx]
    print(f"Predicted class: {pred_name} (idx={pred_idx})")

    save_path = os.path.join(args.output, f"gradcam_{pred_name.replace(' ', '_')}.png")
    plot_gradcam(image_pil, cam, class_name=pred_name, save_path=save_path)
    print(f"GradCAM saved to: {save_path}")

    # ---- prototype attention (if cross-attention is enabled) ----
    if model.use_cross_attn:
        with torch.no_grad():
            img_feats = clip_model.encode_image(image_tensor)
            img_feats = F.normalize(img_feats, dim=-1)
            _, aux = model(img_feats, clip_model)

        if "attn_weights" in aux:
            attn = aux["attn_weights"][0]
            n_text = len(classnames)
            levels = cfg.get("prototype_levels", [4, 8, 16])

            attn_path = os.path.join(args.output, f"proto_attn_{pred_name.replace(' ', '_')}.png")
            plot_prototype_attention(attn, n_text, classnames, levels, save_path=attn_path)
            print(f"Prototype attention saved to: {attn_path}")


if __name__ == "__main__":
    main()
