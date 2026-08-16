"""Evaluate CLIP zero-shot classification on all datasets.

Usage:
    python scripts/evaluate_zeroshot.py --root_path data/ --output results/zeroshot/
    python scripts/evaluate_zeroshot.py --root_path data/ --dataset plantdoc --backbone ViT-B/16
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from tqdm import tqdm

from mvpdr import clip
from mvpdr.datasets import build_dataset
from mvpdr.datasets.base import build_data_loader
from mvpdr.utils import clip_classifier, cls_acc, cls_acc_test


BACKBONES = ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]
DATASETS = ["plantdoc", "plantvillage"]


def evaluate_zeroshot(dataset_name, root_path, backbone, device):
    clip_model, preprocess = clip.load(backbone, device=device)
    clip_model.eval()

    dataset = build_dataset(dataset_name, root_path, 1)
    classnames = dataset.classnames
    n_classes = len(classnames)

    test_loader = build_data_loader(
        dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False,
    )

    clip_weights = clip_classifier(classnames, dataset.template, clip_model)

    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"{dataset_name}/{backbone}"):
            images = images.to(device)
            features = clip_model.encode_image(images)
            features = F.normalize(features, dim=-1)
            all_features.append(features)
            all_labels.append(labels.to(device))

    features = torch.cat(all_features)
    labels = torch.cat(all_labels)

    logits = 100.0 * features @ clip_weights
    labels_range = np.arange(n_classes)
    result = cls_acc_test(logits, labels, labels=labels_range)

    pred = logits.topk(1, 1, True, True)[1].squeeze().cpu().numpy()
    true = labels.cpu().numpy()
    report = classification_report(
        true, pred, target_names=classnames, output_dict=True, zero_division=0,
    )

    return {
        "dataset": dataset_name,
        "backbone": backbone,
        "n_classes": n_classes,
        "n_test": len(labels),
        "accuracy": float(result["acc"]),
        "precision": float(result["precision"]),
        "recall": float(result["recall"]),
        "f1": float(result["f1"]),
        "per_class": report,
    }


def main():
    parser = argparse.ArgumentParser(description="CLIP Zero-Shot Evaluation")
    parser.add_argument("--root_path", required=True, help="Dataset root directory")
    parser.add_argument("--output", default="results/zeroshot", help="Output directory")
    parser.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    parser.add_argument("--backbone", default="all", choices=BACKBONES + ["all"])
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    backbones = BACKBONES if args.backbone == "all" else [args.backbone]

    all_results = []

    for ds in datasets:
        for bb in backbones:
            print(f"\nEvaluating: {ds} / {bb}")
            result = evaluate_zeroshot(ds, args.root_path, bb, device)
            all_results.append(result)

            out_path = os.path.join(args.output, f"{ds}_{bb.replace('/', '-')}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            print(f"  Accuracy: {result['accuracy']:.2f}%  F1: {result['f1']:.2f}%")

    print("\n" + "=" * 70)
    print(f"{'Dataset':<15} {'Backbone':<12} {'Acc':>8} {'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['dataset']:<15} {r['backbone']:<12} "
              f"{r['accuracy']:>7.2f}% {r['precision']:>7.2f}% "
              f"{r['recall']:>7.2f}% {r['f1']:>7.2f}%")

    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
