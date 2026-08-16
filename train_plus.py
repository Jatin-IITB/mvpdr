"""MVPDR+ training script with architecture upgrades.

Trains the upgraded MVPDR model with any combination of:
  - CoOp learnable soft prompts
  - Hierarchical multi-granularity prototype bank
  - Cross-attention fusion

Usage:
    python train_plus.py --config configs/plantdoc_plus.yaml --seed 1
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import classification_report
from torchvision import transforms
from tqdm import tqdm

from mvpdr import clip
from mvpdr.datasets import build_dataset
from mvpdr.datasets.base import build_data_loader
from mvpdr.models import MVPDRPlus
from mvpdr.utils import cls_acc, cls_acc_test, pre_load_features
from train import (
    save_confusion_matrix,
    save_per_class_metrics,
    save_training_curves,
    set_random_seed,
)


def get_arguments():
    parser = argparse.ArgumentParser(description="MVPDR+ Training")
    parser.add_argument("--config", default="configs/plantdoc_plus.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backbone", type=str, default=None,
                        choices=["RN50", "RN101", "ViT-B/32", "ViT-B/16"])
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    return parser.parse_args()


def extract_features(clip_model, loader, device):
    features, labels = [], []
    with torch.no_grad():
        for images, target in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            feats = clip_model.encode_image(images)
            feats = F.normalize(feats, dim=-1)
            features.append(feats)
            labels.append(target.to(device))
    return torch.cat(features), torch.cat(labels)


def main():
    start_time = time.time()
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.backbone is not None:
        cfg["backbone"] = args.backbone
    if args.shots is not None:
        cfg["shots"] = args.shots

    set_random_seed(args.seed)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = (
            f"results/{cfg['dataset']}/{cfg['backbone']}/"
            f"plus_seed{args.seed}_{cfg['shots']}shot"
        )
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join("caches", cfg["dataset"])
    os.makedirs(cache_dir, exist_ok=True)
    cfg["cache_dir"] = cache_dir

    print(f"\nMVPDR+ | {cfg['dataset']} | {cfg['backbone']} | "
          f"{cfg['shots']}-shot | seed={args.seed}")
    print(f"  prompt_learner={cfg.get('use_prompt_learner', True)}  "
          f"proto_bank={cfg.get('use_prototype_bank', True)}  "
          f"cross_attn={cfg.get('use_cross_attn', True)}")

    # ---- CLIP backbone (frozen) ----
    clip_model, preprocess = clip.load(cfg["backbone"])
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # ---- dataset & loaders ----
    dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])
    classnames = dataset.classnames
    n_classes = len(classnames)
    cfg["n_classes"] = n_classes
    labels_range = np.arange(n_classes)

    test_loader = build_data_loader(
        dataset.test, batch_size=32, is_train=False, tfm=preprocess, shuffle=False,
    )
    preprocess_size = preprocess.__dict__["transforms"][0].size
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(
            size=preprocess_size, scale=(0.5, 1),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    train_loader_cache = build_data_loader(
        dataset.train_x, batch_size=32, tfm=train_tfm, is_train=True, shuffle=False,
    )
    train_loader = build_data_loader(
        dataset.train_x, batch_size=32, tfm=train_tfm, is_train=True, shuffle=True,
    )

    # ---- build MVPDR+ model ----
    model = MVPDRPlus(clip_model, classnames, cfg).to(device)

    if model.prototype_bank is not None:
        print("Initializing prototype bank from K-Means…")
        train_feats, train_labels = extract_features(
            clip_model, train_loader_cache, device,
        )
        model.prototype_bank.init_from_features(train_feats, train_labels)
        levels = cfg.get("prototype_levels", [4, 8, 16])
        total_protos = sum(levels) * n_classes
        print(f"  {len(levels)} levels {levels} → {total_protos} prototypes")

    # ---- pre-extract test features ----
    test_features, test_labels = pre_load_features(
        {"load_pre_feat": False, "cache_dir": cache_dir},
        "test", clip_model, test_loader,
    )
    test_labels = test_labels.long()  # ensure int dtype (pre_load_features returns long)

    # ---- optimizer (per-component learning rates) ----
    param_groups = []
    if model.prompt_learner is not None:
        param_groups.append({
            "params": model.prompt_learner.parameters(),
            "lr": cfg.get("lr_prompt", 2e-3),
        })
    if model.prototype_bank is not None:
        param_groups.append({
            "params": model.prototype_bank.parameters(),
            "lr": cfg.get("lr_proto", 1e-3),
        })
    if model.cross_attn is not None:
        cross_params = list(model.cross_attn.parameters())
        if hasattr(model, "logit_scale"):
            cross_params.append(model.logit_scale)
        param_groups.append({
            "params": cross_params,
            "lr": cfg.get("lr_fusion", 5e-4),
        })
    if not param_groups:
        raise ValueError("No trainable components enabled")

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=cfg.get("weight_decay", 0.01),
    )
    total_steps = cfg["train_epoch"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    lambda_v = cfg.get("lambda_v", 0.5)
    lambda_t = cfg.get("lambda_t", 0.5)

    # ---- training loop ----
    best_acc, best_epoch = 0.0, 0
    best_result = {}
    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "test_acc": [], "test_precision": [], "test_recall": [], "test_f1": [],
        "learning_rate": [],
    }

    for epoch in range(cfg["train_epoch"]):
        model.train()
        correct, total = 0, 0
        losses = []

        for images, target in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{cfg['train_epoch']}",
        ):
            images, target = images.to(device), target.to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)

            logits, aux = model(image_features, clip_model)

            loss = F.cross_entropy(logits, target)
            if "visual_logits" in aux:
                loss = loss + lambda_v * F.cross_entropy(aux["visual_logits"], target)
            if "textual_logits" in aux and model.use_cross_attn:
                loss = loss + lambda_t * F.cross_entropy(aux["textual_logits"], target)

            acc = cls_acc(logits, target, labels=labels_range)["acc"]
            correct += acc / 100 * len(target)
            total += len(target)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if model.prototype_bank is not None:
                model.prototype_bank.momentum_update(
                    image_features.detach(), target,
                )

        train_acc = correct / total * 100
        train_loss = sum(losses) / len(losses)
        lr = optimizer.param_groups[0]["lr"]

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(test_features, clip_model)
        result = cls_acc(eval_logits, test_labels, labels=labels_range)

        print(f"  Train: {train_acc:.1f}% loss={train_loss:.4f} | "
              f"Test: {result['acc']:.2f}% F1={result['f1']:.1f}% | LR={lr:.6f}")

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(result["acc"])
        history["test_precision"].append(result["precision"])
        history["test_recall"].append(result["recall"])
        history["test_f1"].append(result["f1"])
        history["learning_rate"].append(lr)

        if result["acc"] > best_acc:
            best_acc = result["acc"]
            best_result = result.copy()
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"  ** New best: {best_acc:.2f}% **")

    print(f"\nBest: {best_acc:.2f}% at epoch {best_epoch}")

    # ---- final evaluation with best checkpoint ----
    model.load_state_dict(torch.load(
        os.path.join(output_dir, "best_model.pth"), weights_only=True,
    ))
    model.eval()
    with torch.no_grad():
        final_logits, _ = model(test_features, clip_model)
    final_result = cls_acc_test(final_logits, test_labels, labels=labels_range)

    pred_labels = final_logits.topk(1, 1, True, True)[1].squeeze().cpu().numpy()
    true_labels = test_labels.cpu().numpy()
    class_report = classification_report(
        true_labels, pred_labels,
        target_names=classnames, output_dict=True, zero_division=0,
    )

    elapsed = time.time() - start_time

    # ---- save artifacts ----
    save_training_curves(history, os.path.join(output_dir, "training_curves.png"))
    save_confusion_matrix(
        np.array(final_result["conf_matrix"].cpu()), classnames,
        os.path.join(output_dir, "confusion_matrix.png"),
    )
    save_per_class_metrics(class_report, os.path.join(output_dir, "per_class_metrics.png"))

    levels = cfg.get("prototype_levels", [4, 8, 16])
    results_json = {
        "model": "MVPDR+",
        "dataset": cfg["dataset"],
        "backbone": cfg["backbone"],
        "seed": args.seed,
        "architecture": {
            "prompt_learner": cfg.get("use_prompt_learner", True),
            "n_ctx": cfg.get("n_ctx", 4),
            "prototype_bank": cfg.get("use_prototype_bank", True),
            "prototype_levels": levels,
            "cross_attention": cfg.get("use_cross_attn", True),
            "n_cross_layers": cfg.get("n_cross_layers", 2),
        },
        "performance": {
            "accuracy": float(best_acc),
            "precision": float(best_result.get("precision", 0)),
            "recall": float(best_result.get("recall", 0)),
            "f1": float(best_result.get("f1", 0)),
            "best_epoch": best_epoch,
        },
        "training": {
            "epochs": cfg["train_epoch"],
            "shots": cfg["shots"],
            "time_seconds": round(elapsed, 1),
        },
        "per_class_metrics": class_report,
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, "training_history.csv"), index=False,
    )

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(f"MVPDR+ Classification Report\n{'=' * 60}\n")
        f.write(f"Dataset: {cfg['dataset']} | Backbone: {cfg['backbone']} | "
                f"Seed: {args.seed}\n")
        f.write(f"Training Time: {elapsed / 60:.1f} min\n")
        f.write(f"Architecture: CoOp({cfg.get('n_ctx', 4)}ctx) + "
                f"HierProto{levels} + CrossAttn\n\n")
        f.write(f"Accuracy:  {best_acc:.2f}%\n")
        f.write(f"Precision: {best_result.get('precision', 0):.2f}%\n")
        f.write(f"Recall:    {best_result.get('recall', 0):.2f}%\n")
        f.write(f"F1-Score:  {best_result.get('f1', 0):.2f}%\n")
        f.write(f"Best Epoch: {best_epoch}/{cfg['train_epoch']}\n\n")
        f.write(f"{'Class':<40} {'P':>6} {'R':>6} {'F1':>6} {'N':>6}\n")
        f.write("-" * 66 + "\n")
        for cls in classnames:
            if cls in class_report:
                m = class_report[cls]
                f.write(
                    f"{cls:<40} {m['precision'] * 100:>5.1f}% "
                    f"{m['recall'] * 100:>5.1f}% "
                    f"{m['f1-score'] * 100:>5.1f}% "
                    f"{int(m['support']):>6}\n"
                )

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "backbone": cfg["backbone"],
        "classnames": classnames,
        "performance": results_json["performance"],
    }, os.path.join(output_dir, "mvpdr_plus_model.pth"))

    print(f"\nResults: Acc={best_acc:.2f}% "
          f"F1={best_result.get('f1', 0):.2f}% ({elapsed / 60:.1f}min)")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
