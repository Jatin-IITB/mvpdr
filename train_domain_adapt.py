"""MVPDR+ domain adaptation training.

Trains MVPDRPlus on a labeled source domain while aligning features to
an unlabeled target domain using CORAL and/or DANN.

Usage:
    python train_domain_adapt.py \\
        --source_config configs/plantvillage_plus.yaml \\
        --target_config configs/plantdoc_plus.yaml \\
        --method coral --seed 1
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import classification_report
from torchvision import transforms
from tqdm import tqdm

from mvpdr import clip
from mvpdr.adaptation import CORAL, DomainDiscriminator, dann_loss, grl_alpha_schedule
from mvpdr.datasets import build_dataset
from mvpdr.datasets.base import build_data_loader
from mvpdr.models import MVPDRPlus
from mvpdr.utils import cls_acc, pre_load_features
from train import set_random_seed


def get_arguments():
    parser = argparse.ArgumentParser(description="MVPDR+ Domain Adaptation")
    parser.add_argument("--source_config", required=True)
    parser.add_argument("--target_config", required=True)
    parser.add_argument("--method", choices=["coral", "dann", "both"], default="coral")
    parser.add_argument("--lambda_coral", type=float, default=1.0)
    parser.add_argument("--lambda_dann", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()


def build_train_transform(preprocess):
    size = preprocess.__dict__["transforms"][0].size
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=size, scale=(0.5, 1),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


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


def infinite_loader(loader):
    while True:
        yield from loader


def main():
    start_time = time.time()
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.source_config) as f:
        src_cfg = yaml.safe_load(f)
    with open(args.target_config) as f:
        tgt_cfg = yaml.safe_load(f)

    set_random_seed(args.seed)

    output_dir = (
        f"results/da_{src_cfg['dataset']}_to_{tgt_cfg['dataset']}/"
        f"{src_cfg['backbone']}/{args.method}_seed{args.seed}"
    )
    os.makedirs(output_dir, exist_ok=True)
    src_cache = os.path.join("caches", src_cfg["dataset"])
    tgt_cache = os.path.join("caches", tgt_cfg["dataset"])
    os.makedirs(src_cache, exist_ok=True)
    os.makedirs(tgt_cache, exist_ok=True)
    src_cfg["cache_dir"] = src_cache
    tgt_cfg["cache_dir"] = tgt_cache

    print(f"\nDomain Adaptation: {src_cfg['dataset']} → {tgt_cfg['dataset']}")
    print(f"Method: {args.method} | Backbone: {src_cfg['backbone']} | Seed: {args.seed}")

    # ---- CLIP backbone ----
    clip_model, preprocess = clip.load(src_cfg["backbone"])
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # ---- source dataset (labeled) ----
    src_dataset = build_dataset(src_cfg["dataset"], src_cfg["root_path"], src_cfg["shots"])
    classnames = src_dataset.classnames
    n_classes = len(classnames)
    src_cfg["n_classes"] = n_classes
    labels_range = np.arange(n_classes)

    train_tfm = build_train_transform(preprocess)
    src_train_loader = build_data_loader(
        src_dataset.train_x, batch_size=32, tfm=train_tfm, is_train=True, shuffle=True,
    )
    src_train_loader_cache = build_data_loader(
        src_dataset.train_x, batch_size=32, tfm=train_tfm, is_train=True, shuffle=False,
    )

    # ---- target dataset (unlabeled for adaptation, labeled for eval) ----
    tgt_dataset = build_dataset(tgt_cfg["dataset"], tgt_cfg["root_path"], tgt_cfg.get("shots", 16))
    tgt_train_loader = build_data_loader(
        tgt_dataset.train_x, batch_size=32, tfm=train_tfm, is_train=True, shuffle=True,
    )
    tgt_test_loader = build_data_loader(
        tgt_dataset.test, batch_size=32, is_train=False, tfm=preprocess, shuffle=False,
    )

    # ---- model ----
    model = MVPDRPlus(clip_model, classnames, src_cfg).to(device)

    if model.prototype_bank is not None:
        print("Initializing prototype bank from source features…")
        src_feats, src_labels = extract_features(clip_model, src_train_loader_cache, device)
        model.prototype_bank.init_from_features(src_feats, src_labels)

    # ---- adaptation modules ----
    use_coral = args.method in ("coral", "both")
    use_dann = args.method in ("dann", "both")

    coral_module = CORAL(lambda_coral=args.lambda_coral) if use_coral else None
    domain_disc = DomainDiscriminator(embed_dim=n_classes).to(device) if use_dann else None

    # ---- optimizer ----
    param_groups = []
    if model.prompt_learner is not None:
        param_groups.append({"params": model.prompt_learner.parameters(), "lr": src_cfg.get("lr_prompt", 2e-3)})
    if model.prototype_bank is not None:
        param_groups.append({"params": model.prototype_bank.parameters(), "lr": src_cfg.get("lr_proto", 1e-3)})
    if model.cross_attn is not None:
        cross_params = list(model.cross_attn.parameters())
        if hasattr(model, "logit_scale"):
            cross_params.append(model.logit_scale)
        param_groups.append({"params": cross_params, "lr": src_cfg.get("lr_fusion", 5e-4)})
    if domain_disc is not None:
        param_groups.append({"params": domain_disc.parameters(), "lr": 1e-3})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=src_cfg.get("weight_decay", 0.01))
    epochs = src_cfg.get("train_epoch", 30)
    total_steps = epochs * len(src_train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    lambda_v = src_cfg.get("lambda_v", 0.5)
    lambda_t = src_cfg.get("lambda_t", 0.5)

    # ---- pre-extract target test features ----
    tgt_test_features, tgt_test_labels = pre_load_features(
        {"load_pre_feat": False, "cache_dir": tgt_cache},
        "test", clip_model, tgt_test_loader,
    )
    tgt_test_labels = tgt_test_labels.long()

    # ---- training loop ----
    best_acc, best_epoch = 0.0, 0
    history = {"epoch": [], "src_loss": [], "da_loss": [], "tgt_acc": []}
    tgt_iter = iter(infinite_loader(tgt_train_loader))

    for epoch in range(epochs):
        model.train()
        if domain_disc is not None:
            domain_disc.train()

        epoch_src_loss, epoch_da_loss = [], []
        alpha = grl_alpha_schedule(epoch, epochs) if use_dann else 0.0

        for src_images, src_target in tqdm(
            src_train_loader, desc=f"Epoch {epoch + 1}/{epochs}",
        ):
            src_images, src_target = src_images.to(device), src_target.to(device)
            tgt_images, _ = next(tgt_iter)
            tgt_images = tgt_images.to(device)

            with torch.no_grad():
                src_feats = F.normalize(clip_model.encode_image(src_images), dim=-1)
                tgt_feats = F.normalize(clip_model.encode_image(tgt_images), dim=-1)

            # ---- source classification loss ----
            src_logits, aux = model(src_feats, clip_model)
            loss_cls = F.cross_entropy(src_logits, src_target)
            if "visual_logits" in aux:
                loss_cls = loss_cls + lambda_v * F.cross_entropy(aux["visual_logits"], src_target)
            if "textual_logits" in aux and model.use_cross_attn:
                loss_cls = loss_cls + lambda_t * F.cross_entropy(aux["textual_logits"], src_target)

            # ---- target forward pass (for DA — needs gradient through model) ----
            tgt_logits, _ = model(tgt_feats, clip_model)

            # ---- domain adaptation losses ----
            # DA operates on model OUTPUT (logits), not raw CLIP features,
            # so gradients flow back through prompt learner / prototype bank /
            # cross-attention — the actual adaptation targets.
            loss_da = torch.tensor(0.0, device=device)

            if use_coral and coral_module is not None:
                loss_da = loss_da + coral_module(src_logits, tgt_logits)

            if use_dann and domain_disc is not None:
                all_logits = torch.cat([src_logits, tgt_logits], dim=0)
                domain_labels = torch.cat([
                    torch.ones(src_logits.size(0), device=device),
                    torch.zeros(tgt_logits.size(0), device=device),
                ])
                domain_preds = domain_disc(all_logits, alpha=alpha)
                loss_da = loss_da + args.lambda_dann * dann_loss(domain_preds, domain_labels)

            loss = loss_cls + loss_da
            epoch_src_loss.append(loss_cls.item())
            epoch_da_loss.append(loss_da.item())

            optimizer.zero_grad()
            loss.backward()
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if model.prototype_bank is not None:
                model.prototype_bank.momentum_update(src_feats.detach(), src_target)

        # ---- evaluate on target test set ----
        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(tgt_test_features, clip_model)
        result = cls_acc(eval_logits, tgt_test_labels, labels=labels_range)
        tgt_acc = result["acc"]

        mean_src = sum(epoch_src_loss) / len(epoch_src_loss)
        mean_da = sum(epoch_da_loss) / len(epoch_da_loss)
        print(f"  Source loss={mean_src:.4f}  DA loss={mean_da:.4f}  "
              f"Target acc={tgt_acc:.2f}%")

        history["epoch"].append(epoch + 1)
        history["src_loss"].append(mean_src)
        history["da_loss"].append(mean_da)
        history["tgt_acc"].append(tgt_acc)

        if tgt_acc > best_acc:
            best_acc = tgt_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"  ** New best: {best_acc:.2f}% **")

    elapsed = time.time() - start_time

    # ---- save results ----
    import pandas as pd
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "history.csv"), index=False)

    results_json = {
        "method": args.method,
        "source": src_cfg["dataset"],
        "target": tgt_cfg["dataset"],
        "backbone": src_cfg["backbone"],
        "seed": args.seed,
        "target_accuracy": float(best_acc),
        "best_epoch": best_epoch,
        "lambda_coral": args.lambda_coral if use_coral else None,
        "lambda_dann": args.lambda_dann if use_dann else None,
        "time_seconds": round(elapsed, 1),
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nBest target accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    print(f"Training time: {elapsed / 60:.1f} min")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
