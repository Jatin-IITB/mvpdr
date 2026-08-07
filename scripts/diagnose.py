"""Full diagnostic pipeline — classification + severity + treatment report.

Runs MVPDR+ inference, estimates severity from GradCAM, queries the
knowledge base, and optionally invokes the Claude agent for a rich
diagnostic report.

Usage:
    # Rule-based (no API key needed):
    python scripts/diagnose.py --config configs/plantdoc_plus.yaml \\
        --image path/to/leaf.jpg

    # Agentic mode (requires ANTHROPIC_API_KEY):
    ANTHROPIC_API_KEY=sk-... python scripts/diagnose.py \\
        --config configs/plantdoc_plus.yaml \\
        --checkpoint results/.../model.pth \\
        --image path/to/leaf.jpg \\
        --output report.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mvpdr import clip
from mvpdr.agent import DiagnosticContext, run_agentic_diagnosis
from mvpdr.interpretability import GradCAM
from mvpdr.knowledge import DiseaseKnowledgeBase
from mvpdr.models import MVPDRPlus
from mvpdr.severity import estimate_severity_heuristic
from mvpdr.utils import clip_classifier


def main():
    parser = argparse.ArgumentParser(description="MVPDR+ Diagnostic Pipeline")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint")
    parser.add_argument("--image", required=True, help="Input leaf image path")
    parser.add_argument("--output", default=None, help="Save report JSON to file")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--zero_shot", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    clip_model, preprocess = clip.load(cfg["backbone"])
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    if args.checkpoint and not args.zero_shot:
        ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
        classnames = ckpt.get("classnames")
        if classnames is None:
            from mvpdr.datasets import build_dataset
            dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])
            classnames = dataset.classnames
        cfg["n_classes"] = len(classnames)
        model = MVPDRPlus(clip_model, classnames, cfg).to(device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.eval()
    else:
        from mvpdr.datasets import build_dataset
        dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])
        classnames = dataset.classnames
        cfg["n_classes"] = len(classnames)
        model = None

    # ---- inference ----
    image_pil = Image.open(args.image).convert("RGB")
    image_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features = F.normalize(image_features, dim=-1)

    template = ["a photo of a {}."]
    if model is not None:
        with torch.no_grad():
            logits, aux = model(image_features, clip_model)
            if model.prompt_learner is not None:
                text_features = model.prompt_learner(clip_model)
                text_features = F.normalize(text_features, dim=-1)
            else:
                text_features = clip_classifier(classnames, template, clip_model).t()
    else:
        text_weights = clip_classifier(classnames, template, clip_model)
        logits = 100.0 * image_features.float() @ text_weights.float()
        text_features = text_weights.t()

    probs = F.softmax(logits, dim=-1).squeeze(0)
    top_k = min(args.top_k, len(classnames))
    top_probs, top_idx = torch.topk(probs, top_k)

    top_predictions = [
        (classnames[idx.item()], prob.item())
        for idx, prob in zip(top_idx, top_probs)
    ]

    print(f"Top predictions:")
    for name, conf in top_predictions:
        print(f"  {name}: {conf:.1%}")

    # ---- GradCAM + severity ----
    gradcam = GradCAM(clip_model)
    try:
        cam, _ = gradcam.generate(image_tensor, text_features, class_idx=top_idx[0].item())
        cam_resized = np.array(
            Image.fromarray(np.uint8(cam * 255)).resize(
                image_pil.size, resample=Image.BILINEAR,
            )
        ) / 255.0
        severity = estimate_severity_heuristic(
            gradcam_map=cam_resized,
            classification_confidence=top_probs[0].item(),
        )
        print(f"\nSeverity: {severity.label} (score={severity.score:.3f})")
        print(f"  GradCAM coverage: {severity.gradcam_coverage:.1%}")
    except RuntimeError as e:
        print(f"\nGradCAM failed: {e}")
        cam = None
        severity = None
    finally:
        gradcam.remove_hooks()

    # ---- diagnostic report ----
    kb = DiseaseKnowledgeBase()

    ctx = DiagnosticContext(
        classnames=classnames,
        top_predictions=top_predictions,
        severity=severity,
        knowledge_base=kb,
    )

    print("\nGenerating diagnostic report...")
    report = run_agentic_diagnosis(ctx)

    print(f"\n{'=' * 60}")
    print(report.to_markdown())
    print(f"{'=' * 60}")

    if args.output:
        with open(args.output, "w") as f:
            f.write(report.to_json())
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
