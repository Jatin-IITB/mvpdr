"""Temperature scaling calibration for MVPDR+.

Splits the test set into validation (for fitting T) and test (for evaluation),
then produces reliability diagrams and ECE metrics before/after calibration.

Usage:
    python scripts/calibrate.py \\
        --config configs/plantdoc_plus.yaml \\
        --checkpoint results/.../best_model.pth \\
        --output calibration_results/
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mvpdr import clip
from mvpdr.calibration import (
    TemperatureScaling,
    expected_calibration_error,
    reliability_diagram,
)
from mvpdr.datasets import build_dataset
from mvpdr.datasets.base import build_data_loader
from mvpdr.models import MVPDRPlus
from mvpdr.utils import pre_load_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="calibration_results")
    parser.add_argument("--val_fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output, exist_ok=True)
    cache_dir = os.path.join("caches", cfg["dataset"])
    os.makedirs(cache_dir, exist_ok=True)
    cfg["cache_dir"] = cache_dir

    # ---- load model ----
    clip_model, preprocess = clip.load(cfg["backbone"])
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
    classnames = ckpt.get("classnames")

    if classnames is None:
        dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])
        classnames = dataset.classnames

    cfg["n_classes"] = len(classnames)
    model = MVPDRPlus(clip_model, classnames, cfg).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    # ---- load test features ----
    dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])
    test_loader = build_data_loader(
        dataset.test, batch_size=32, is_train=False, tfm=preprocess, shuffle=False,
    )
    test_features, test_labels = pre_load_features(
        {"load_pre_feat": False, "cache_dir": cache_dir},
        "test", clip_model, test_loader,
    )
    test_labels = test_labels.long()

    # ---- compute logits ----
    with torch.no_grad():
        logits, _ = model(test_features, clip_model)
    logits = logits.cpu()
    test_labels = test_labels.cpu()

    # ---- split into val / test ----
    n = logits.shape[0]
    perm = torch.randperm(n)
    n_val = int(n * args.val_fraction)
    val_idx, test_idx = perm[:n_val], perm[n_val:]

    val_logits, val_labels = logits[val_idx], test_labels[val_idx]
    test_logits, test_labels_split = logits[test_idx], test_labels[test_idx]

    print(f"Samples: {n_val} validation, {n - n_val} test")

    # ---- before calibration ----
    uncal_probs = F.softmax(test_logits, dim=-1)
    ece_before = expected_calibration_error(uncal_probs, test_labels_split)
    acc_before = (uncal_probs.argmax(dim=-1) == test_labels_split).float().mean().item()

    print(f"\nBefore calibration:")
    print(f"  Accuracy: {acc_before * 100:.2f}%")
    print(f"  ECE:      {ece_before:.4f}")

    reliability_diagram(
        uncal_probs, test_labels_split,
        save_path=os.path.join(args.output, "reliability_before.png"),
    )

    # ---- fit temperature ----
    ts = TemperatureScaling()
    T = ts.fit(val_logits, val_labels)
    print(f"\nLearned temperature: {T:.4f}")

    # ---- after calibration ----
    cal_probs = ts.calibrated_probs(test_logits)
    ece_after = expected_calibration_error(cal_probs, test_labels_split)
    acc_after = (cal_probs.argmax(dim=-1) == test_labels_split).float().mean().item()

    print(f"\nAfter calibration (T={T:.4f}):")
    print(f"  Accuracy: {acc_after * 100:.2f}%")
    print(f"  ECE:      {ece_after:.4f}")

    reliability_diagram(
        cal_probs, test_labels_split,
        save_path=os.path.join(args.output, "reliability_after.png"),
    )

    # ---- save results ----
    results = {
        "dataset": cfg["dataset"],
        "backbone": cfg["backbone"],
        "temperature": T,
        "val_samples": n_val,
        "test_samples": n - n_val,
        "before": {"accuracy": acc_before, "ece": ece_before},
        "after": {"accuracy": acc_after, "ece": ece_after},
        "ece_reduction": ece_before - ece_after,
    }
    with open(os.path.join(args.output, "calibration_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    pct = (ece_before - ece_after) / max(ece_before, 1e-8) * 100
    print(f"\nECE reduction: {ece_before:.4f} → {ece_after:.4f} "
          f"({pct:.1f}% improvement)")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
