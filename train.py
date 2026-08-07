import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import classification_report
from torchvision import transforms
from tqdm import tqdm

from mvpdr import clip
from mvpdr.datasets import build_dataset
from mvpdr.datasets.base import build_data_loader
from mvpdr.utils import (
    build_textual_prototypes,
    build_visual_prototypes,
    clip_classifier,
    cls_acc,
    cls_acc_test,
    pre_load_features,
)

PROMPT_PATHS = {
    "plantwild": "prompts/plantwild_prompts_50_18.json",
    "plantdoc": "prompts/plantdoc_prompts_50_25.json",
    "plantvillage": "prompts/plantvillage_prompts_50_25.json",
}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_arguments():
    parser = argparse.ArgumentParser(description="MVPDR - Multi-View Prototype-based Disease Recognition")
    parser.add_argument("--config", default="configs/plantdoc.yaml", help="Path to config file")
    parser.add_argument("--nclt", type=int, default=16, help="Number of clusters for visual prototypes")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--backbone", type=str, default="RN101", choices=["RN50", "RN101", "ViT-B/32", "ViT-B/16"])
    parser.add_argument("--w1", type=float, default=1.0, help="Weight for visual prototype loss")
    parser.add_argument("--w2", type=float, default=0.1, help="Weight for textual max loss")
    parser.add_argument("--w3", type=float, default=0.1, help="Weight for textual mean loss")
    parser.add_argument("--alpha", type=float, default=0.3, help="Visual-textual fusion weight")
    parser.add_argument("--bbeta", type=float, default=0.5, help="Beta for prototype affinity")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma for textual fusion")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID")
    return parser.parse_args()


def compute_logits(image_features, adapter, prompt_adapter, v_labels, n_class, alpha, bbeta, gamma):
    affinity = adapter(image_features)
    v_logits = ((-1) * (bbeta - bbeta * affinity)).exp() @ v_labels

    t_logits = 100.0 * prompt_adapter(image_features)
    t_logits = t_logits.reshape(t_logits.shape[0], n_class, -1)
    t_mean_logits = t_logits.mean(dim=-1)
    t_max_logits = t_logits.max(dim=-1)[0]
    t_logits = gamma * t_mean_logits + bbeta * t_max_logits

    mvpdr_logits = t_logits + v_logits * alpha
    return mvpdr_logits, v_logits, t_mean_logits, t_max_logits


def run_mvpdr(cfg, v_prototypes, v_labels, test_features, test_labels,
              textual_prototypes, clip_model, train_loader_F, weights, class_names):
    device = next(clip_model.parameters()).device
    n_class = v_labels.shape[-1]
    gamma, bbeta, alpha = cfg["gamma"], cfg["bbeta"], cfg["alpha"]
    labels = np.arange(n_class)

    adapter = nn.Linear(v_prototypes.shape[0], v_prototypes.shape[1], bias=False).to(clip_model.dtype).to(device)
    adapter.weight = nn.Parameter(v_prototypes.t())

    prompt_adapter = nn.Linear(textual_prototypes.shape[0], textual_prototypes.shape[1], bias=False).to(clip_model.dtype).to(device)
    prompt_adapter.weight = nn.Parameter(textual_prototypes.t())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["lr"], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["train_epoch"] * len(train_loader_F))

    prompt_optimizer = torch.optim.AdamW(prompt_adapter.parameters(), lr=cfg["lr"], eps=1e-4)
    prompt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(prompt_optimizer, cfg["train_epoch"] * len(train_loader_F))

    best_acc, best_epoch = 0.0, 0
    train_history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "test_acc": [], "test_precision": [], "test_recall": [], "test_f1": [],
        "learning_rate": [],
    }

    for epoch in range(cfg["train_epoch"]):
        adapter.train()
        prompt_adapter.train()
        correct, total = 0, 0
        losses = []

        for images, target in tqdm(train_loader_F, desc=f"Epoch {epoch + 1}/{cfg['train_epoch']}"):
            images, target = images.to(device), target.to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            mvpdr_logits, v_logits, t_mean_logits, t_max_logits = compute_logits(
                image_features, adapter, prompt_adapter, v_labels, n_class, alpha, bbeta, gamma
            )

            w1, w2, w3 = weights
            loss = w1 * F.cross_entropy(v_logits, target) + w2 * F.cross_entropy(t_max_logits, target) + w3 * F.cross_entropy(t_mean_logits, target)

            acc = cls_acc(mvpdr_logits, target, labels=labels)["acc"]
            correct += acc / 100 * len(mvpdr_logits)
            total += len(mvpdr_logits)
            losses.append(loss.item())

            optimizer.zero_grad()
            prompt_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(prompt_adapter.parameters(), max_norm=1.0)
            optimizer.step()
            prompt_optimizer.step()
            scheduler.step()
            prompt_scheduler.step()

        train_acc = correct / total * 100
        train_loss = sum(losses) / len(losses)
        lr = scheduler.get_last_lr()[0]

        adapter.eval()
        prompt_adapter.eval()
        with torch.no_grad():
            mvpdr_logits, _, _, _ = compute_logits(
                test_features, adapter, prompt_adapter, v_labels, n_class, alpha, bbeta, gamma
            )
        result = cls_acc(mvpdr_logits, test_labels, labels=labels)
        acc = result["acc"]

        print(f"  LR: {lr:.6f} | Train: {train_acc:.1f}% loss={train_loss:.4f} | "
              f"Test: {acc:.2f}% P={result['precision']:.1f} R={result['recall']:.1f} F1={result['f1']:.1f}")

        train_history["epoch"].append(epoch + 1)
        train_history["train_loss"].append(train_loss)
        train_history["train_acc"].append(train_acc)
        train_history["test_acc"].append(acc)
        train_history["test_precision"].append(result["precision"])
        train_history["test_recall"].append(result["recall"])
        train_history["test_f1"].append(result["f1"])
        train_history["learning_rate"].append(lr)

        if acc > best_acc:
            best_acc = acc
            best_precision, best_recall, best_f1 = result["precision"], result["recall"], result["f1"]
            best_epoch = epoch + 1
            torch.save(adapter.weight, os.path.join(cfg["cache_dir"], f"best_F_{cfg['shots']}shots.pt"))
            torch.save(prompt_adapter.weight, os.path.join(cfg["cache_dir"], "best_prompt.pt"))
            print(f"  ** New best: {best_acc:.2f}% **")

    print(f"\nBest accuracy: {best_acc:.2f}% at epoch {best_epoch}")

    adapter.weight = torch.load(os.path.join(cfg["cache_dir"], f"best_F_{cfg['shots']}shots.pt"))
    prompt_adapter.weight = torch.load(os.path.join(cfg["cache_dir"], "best_prompt.pt"))
    adapter.eval()
    prompt_adapter.eval()

    with torch.no_grad():
        mvpdr_logits, _, _, _ = compute_logits(
            test_features, adapter, prompt_adapter, v_labels, n_class, alpha, bbeta, gamma
        )
    final_result = cls_acc_test(mvpdr_logits, test_labels, labels=labels)

    pred_labels = mvpdr_logits.topk(1, 1, True, True)[1].squeeze().cpu().numpy()
    true_labels = test_labels.cpu().numpy()
    class_report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True, zero_division=0)

    return {
        "best_acc": best_acc, "best_precision": best_precision,
        "best_recall": best_recall, "best_f1": best_f1, "best_epoch": best_epoch,
        "train_history": train_history,
        "conf_matrix": np.array(final_result["conf_matrix"].cpu()),
        "class_report": class_report,
        "predictions": pred_labels, "true_labels": true_labels,
    }


def save_training_curves(history, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history["epoch"], history["train_loss"], "b-", lw=2)
    axes[0, 0].set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history["epoch"], history["train_acc"], "g-", lw=2)
    axes[0, 1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Training Accuracy")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history["epoch"], history["test_acc"], "r-", lw=2, label="Accuracy")
    axes[1, 0].plot(history["epoch"], history["test_f1"], "b--", lw=2, label="F1")
    axes[1, 0].set(xlabel="Epoch", ylabel="Score (%)", title="Test Performance")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history["epoch"], history["learning_rate"], color="purple", lw=2)
    axes[1, 1].set(xlabel="Epoch", ylabel="LR", title="Learning Rate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(conf_matrix, class_names, path):
    conf_norm = conf_matrix.astype("float") / conf_matrix.sum(axis=1, keepdims=True)
    plt.figure(figsize=(max(12, len(class_names) * 0.4), max(10, len(class_names) * 0.35)))
    sns.heatmap(conf_norm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Normalized)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_per_class_metrics(report, path):
    classes = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
    if not classes:
        return
    x = np.arange(len(classes))
    metrics_data = {
        "Precision": [report[c]["precision"] * 100 for c in classes],
        "Recall": [report[c]["recall"] * 100 for c in classes],
        "F1-Score": [report[c]["f1-score"] * 100 for c in classes],
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = ["skyblue", "lightcoral", "lightgreen"]
    for ax, (name, values), color in zip(axes, metrics_data.items(), colors):
        ax.bar(x, values, color=color, alpha=0.8)
        ax.axhline(y=np.mean(values), color="r", ls="--", label=f"Mean: {np.mean(values):.1f}%")
        ax.set(ylabel=f"{name} (%)", title=f"{name} by Class")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    start_time = time.time()
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert os.path.exists(args.config), f"Config not found: {args.config}"
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cache_dir = os.path.join("caches", cfg["dataset"])
    os.makedirs(cache_dir, exist_ok=True)
    cfg["cache_dir"] = cache_dir
    cfg["backbone"] = args.backbone
    cfg["weights"] = [args.w1, args.w2, args.w3]
    cfg["alpha"], cfg["bbeta"], cfg["gamma"] = args.alpha, args.bbeta, args.gamma

    output_dir = f"results/{cfg['dataset']}/{args.backbone}/seed{args.seed}_alpha{args.alpha}_nclt{args.nclt}"
    os.makedirs(output_dir, exist_ok=True)

    set_random_seed(args.seed)

    print(f"\nMVPDR | {cfg['dataset']} | {args.backbone} | {cfg['shots']}-shot | seed={args.seed}")
    print(f"  alpha={args.alpha} bbeta={args.bbeta} gamma={args.gamma} nclt={args.nclt}")

    clip_model, preprocess = clip.load(cfg["backbone"])
    clip_model.eval()
    preprocess_size = preprocess.__dict__["transforms"][0].size

    dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])
    test_loader = build_data_loader(dataset.test, batch_size=32, is_train=False, tfm=preprocess, shuffle=False)

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(size=preprocess_size, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    train_loader_cache = build_data_loader(dataset.train_x, batch_size=32, tfm=train_tfm, is_train=True, shuffle=False)
    train_loader_F = build_data_loader(dataset.train_x, batch_size=32, tfm=train_tfm, is_train=True, shuffle=True)

    if cfg["dataset"] in PROMPT_PATHS:
        classnames = dataset.origin_classes if cfg["dataset"] == "plantvillage" else dataset.classnames
        textual_prototypes = build_textual_prototypes(classnames, dataset.template, clip_model, PROMPT_PATHS[cfg["dataset"]])
    else:
        textual_prototypes = clip_classifier(dataset.classnames, dataset.template, clip_model)

    v_proto_path = os.path.join(cache_dir, f"v_prototypes_{cfg['shots']}shots.pt")
    v_labels_path = os.path.join(cache_dir, f"v_labels_{cfg['shots']}shots.pt")
    if cfg["load_cache"] and os.path.exists(v_proto_path):
        v_prototypes = torch.load(v_proto_path)
        v_labels = torch.load(v_labels_path)
    else:
        v_prototypes, v_labels = build_visual_prototypes(cfg, clip_model, train_loader_cache, len(dataset.classnames), n_clt=args.nclt)
        torch.save(v_prototypes, v_proto_path)
        torch.save(v_labels, v_labels_path)

    test_feat_path = os.path.join(cache_dir, "test_features.pt")
    test_lbl_path = os.path.join(cache_dir, "test_labels.pt")
    if cfg["load_pre_feat"] and os.path.exists(test_feat_path):
        test_features = torch.load(test_feat_path)
        test_labels = torch.load(test_lbl_path)
    else:
        test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
        torch.save(test_features, test_feat_path)
        torch.save(test_labels, test_lbl_path)

    results = run_mvpdr(cfg, v_prototypes, v_labels, test_features, test_labels,
                        textual_prototypes, clip_model, train_loader_F, cfg["weights"], dataset.classnames)

    elapsed = time.time() - start_time

    save_training_curves(results["train_history"], os.path.join(output_dir, "training_curves.png"))
    save_confusion_matrix(results["conf_matrix"], dataset.classnames, os.path.join(output_dir, "confusion_matrix.png"))
    save_per_class_metrics(results["class_report"], os.path.join(output_dir, "per_class_metrics.png"))

    results_json = {
        "dataset": cfg["dataset"],
        "backbone": args.backbone,
        "seed": args.seed,
        "hyperparameters": {
            "nclt": args.nclt, "w1": args.w1, "w2": args.w2, "w3": args.w3,
            "alpha": args.alpha, "bbeta": args.bbeta, "gamma": args.gamma,
            "lr": cfg["lr"], "epochs": cfg["train_epoch"], "shots": cfg["shots"],
        },
        "performance": {
            "accuracy": float(results["best_acc"]),
            "precision": float(results["best_precision"]),
            "recall": float(results["best_recall"]),
            "f1": float(results["best_f1"]),
            "best_epoch": int(results["best_epoch"]),
        },
        "training_time_seconds": round(elapsed, 1),
        "per_class_metrics": results["class_report"],
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    pd.DataFrame(results["train_history"]).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(f"MVPDR Classification Report\n{'='*60}\n")
        f.write(f"Dataset: {cfg['dataset']} | Backbone: {args.backbone} | Seed: {args.seed}\n")
        f.write(f"Training Time: {elapsed/60:.1f} min\n\n")
        f.write(f"Accuracy:  {results['best_acc']:.2f}%\n")
        f.write(f"Precision: {results['best_precision']:.2f}%\n")
        f.write(f"Recall:    {results['best_recall']:.2f}%\n")
        f.write(f"F1-Score:  {results['best_f1']:.2f}%\n")
        f.write(f"Best Epoch: {results['best_epoch']}/{cfg['train_epoch']}\n\n")
        f.write(f"{'Class':<40} {'P':>6} {'R':>6} {'F1':>6} {'N':>6}\n")
        f.write("-" * 66 + "\n")
        for cls in dataset.classnames:
            if cls in results["class_report"]:
                m = results["class_report"][cls]
                f.write(f"{cls:<40} {m['precision']*100:>5.1f}% {m['recall']*100:>5.1f}% "
                        f"{m['f1-score']*100:>5.1f}% {int(m['support']):>6}\n")

    torch.save({
        "adapter_weight": torch.load(os.path.join(cache_dir, f"best_F_{cfg['shots']}shots.pt")),
        "prompt_weight": torch.load(os.path.join(cache_dir, "best_prompt.pt")),
        "clip_backbone": cfg["backbone"],
        "config": cfg,
        "performance": results_json["performance"],
    }, os.path.join(output_dir, "mvpdr_model.pth"))

    print(f"\nResults: Acc={results['best_acc']:.2f}% F1={results['best_f1']:.2f}% ({elapsed/60:.1f}min)")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
