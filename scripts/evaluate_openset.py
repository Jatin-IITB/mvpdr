"""Open-set detection evaluation for MVPDR+.

Holds out a subset of classes as "unknown" and evaluates how well the model
distinguishes known vs. unknown samples using MSP, Energy, and Mahalanobis.

Usage:
    python scripts/evaluate_openset.py \\
        --config configs/plantdoc_plus.yaml \\
        --checkpoint results/.../best_model.pth \\
        --holdout_ratio 0.2 --seed 1
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
from mvpdr.datasets import build_dataset
from mvpdr.datasets.base import build_data_loader
from mvpdr.models import MVPDRPlus
from mvpdr.openset import (
    energy_score,
    evaluate_openset,
    fit_mahalanobis,
    mahalanobis_score,
    msp_score,
    plot_openset_roc,
    save_openset_results,
)
from mvpdr.utils import pre_load_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--holdout_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="openset_results")
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    n_classes = len(classnames)
    cfg["n_classes"] = n_classes
    model = MVPDRPlus(clip_model, classnames, cfg).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    # ---- split classes into known / unknown ----
    n_holdout = max(1, int(n_classes * args.holdout_ratio))
    perm = np.random.permutation(n_classes)
    unknown_classes = set(perm[:n_holdout].tolist())
    known_classes = set(perm[n_holdout:].tolist())

    print(f"Classes: {n_classes} total, {len(known_classes)} known, {len(unknown_classes)} unknown")
    print(f"Unknown: {[classnames[c] for c in sorted(unknown_classes)]}")

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

    # ---- split features ----
    known_mask = torch.tensor([l.item() in known_classes for l in test_labels])
    unknown_mask = ~known_mask

    known_features = test_features[known_mask]
    known_labels = test_labels[known_mask]
    unknown_features = test_features[unknown_mask]

    print(f"Samples: {known_mask.sum().item()} known, {unknown_mask.sum().item()} unknown")

    # ---- compute logits ----
    with torch.no_grad():
        known_logits, _ = model(known_features, clip_model)
        unknown_logits, _ = model(unknown_features, clip_model)

    # ---- evaluate each method ----
    all_results = {}

    # MSP
    msp_in = msp_score(known_logits)
    msp_out = msp_score(unknown_logits)
    msp_res = evaluate_openset(msp_in, msp_out)
    all_results["MSP"] = msp_res
    save_openset_results(msp_res, args.output, "msp")
    print(f"\nMSP:    AUROC={msp_res['auroc']:.4f}  AUPR={msp_res['aupr']:.4f}  "
          f"FPR@95={msp_res['fpr_at_95tpr']:.4f}")

    # Energy
    energy_in = energy_score(known_logits)
    energy_out = energy_score(unknown_logits)
    energy_res = evaluate_openset(energy_in, energy_out)
    all_results["Energy"] = energy_res
    save_openset_results(energy_res, args.output, "energy")
    print(f"Energy: AUROC={energy_res['auroc']:.4f}  AUPR={energy_res['aupr']:.4f}  "
          f"FPR@95={energy_res['fpr_at_95tpr']:.4f}")

    # Mahalanobis (requires fitting on known training features)
    train_loader = build_data_loader(
        dataset.train_x, batch_size=32, tfm=preprocess, is_train=False, shuffle=False,
    )
    train_features, train_labels = pre_load_features(
        {"load_pre_feat": False, "cache_dir": cache_dir},
        "train", clip_model, train_loader,
    )
    train_labels = train_labels.long()

    train_known_mask = torch.tensor([l.item() in known_classes for l in train_labels])
    train_known_features = train_features[train_known_mask]
    train_known_labels = train_labels[train_known_mask]

    # Remap known labels to contiguous [0, n_known) range so fit_mahalanobis
    # doesn't create empty centroids for held-out classes
    known_class_list = sorted(known_classes)
    label_map = {c: i for i, c in enumerate(known_class_list)}
    train_known_labels_mapped = torch.tensor(
        [label_map[l.item()] for l in train_known_labels]
    )
    known_labels_mapped = torch.tensor(
        [label_map[l.item()] for l in known_labels]
    )

    class_means, precision = fit_mahalanobis(
        train_known_features, train_known_labels_mapped, len(known_classes),
    )
    maha_in = mahalanobis_score(known_features.float(), class_means, precision)
    maha_out = mahalanobis_score(unknown_features.float(), class_means, precision)
    maha_res = evaluate_openset(maha_in, maha_out)
    all_results["Mahalanobis"] = maha_res
    save_openset_results(maha_res, args.output, "mahalanobis")
    print(f"Maha:   AUROC={maha_res['auroc']:.4f}  AUPR={maha_res['aupr']:.4f}  "
          f"FPR@95={maha_res['fpr_at_95tpr']:.4f}")

    # ---- ROC plot ----
    roc_path = os.path.join(args.output, "openset_roc_comparison.png")
    plot_openset_roc(all_results, save_path=roc_path)
    print(f"\nROC plot saved to: {roc_path}")

    # ---- summary ----
    summary = {
        method: {k: v for k, v in res.items() if k not in ("fpr", "tpr", "thresholds")}
        for method, res in all_results.items()
    }
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
