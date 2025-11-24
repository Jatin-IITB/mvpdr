import os
import sys
import random
import argparse
import json
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from domain_adaptation import CORAL
from utils import cls_acc

warnings.filterwarnings('ignore')


def compute_mvpdr_logits(image_features, adapter_weight, prompt_weight, v_labels,
                          alpha=0.3, bbeta=0.5, gamma=0.5):
    """
    Compute MVPDR logits by combining visual and textual features
    using weighted fusion of mean & max logits.
    """
    batch_size = image_features.shape[0]
    num_classes = v_labels.shape[1]

    affinity = image_features @ adapter_weight
    v_logits = ((-1) * (bbeta - bbeta * affinity)).exp() @ v_labels

    t_logits = 100. * (image_features @ prompt_weight)
    t_logits = t_logits.reshape(batch_size, num_classes, -1)
    t_mean_logits = t_logits.mean(dim=-1)
    t_max_logits = t_logits.max(dim=-1)[0]

    t_logits = gamma * t_mean_logits + bbeta * t_max_logits

    return t_logits + v_logits * alpha, v_logits, t_mean_logits, t_max_logits


def train_with_domain_adaptation(args):
    """
    Train MVPDR with Domain Adaptation (PlantVillage -> PlantDoc/PlantWild)
    using CORAL for feature alignment.
    """
    print("\n" + "=" * 80)
    print(f"MVPDR WITH DOMAIN ADAPTATION: {args.source} → {args.target}")
    print("=" * 80)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP backbone
    clip_model, preprocess = clip.load(args.backbone, device)
    print(f"Loaded CLIP backbone: {args.backbone}\n")

    # Load datasets
    print(f"Loading SOURCE dataset: {args.source}")
    source_dataset = build_dataset(args.source, args.data_root, shots=0)
    source_train_loader = build_data_loader(
        source_dataset.train_x, batch_size=32, is_train=True, shuffle=True
    )

    print(f"Loading TARGET dataset: {args.target}")
    target_dataset = build_dataset(args.target, args.data_root, shots=0)
    target_train_loader = build_data_loader(
        target_dataset.train_x, batch_size=32, is_train=True, shuffle=True
    )

    num_classes = len(target_dataset.classnames)
    print(f"\n# Classes: {num_classes}")
    print(f"Source train samples: {len(source_train_loader.dataset)}")
    print(f"Target train samples: {len(target_train_loader.dataset)}")
    print(f"Target test samples: {len(target_dataset.test)}\n")

    # Load prototypes and labels
    cache_dir = os.path.join('caches', args.target)
    v_prototypes = torch.load(os.path.join(cache_dir, 'v_prototypes_0shots.pt'), weights_only=False)
    v_labels     = torch.load(os.path.join(cache_dir, 'v_labels_0shots.pt'), weights_only=False)

    # Load textual prompts
    import json

    prompt_dir = os.path.join(os.path.dirname(__file__), "gpt_files_plt44")
    prompt_file = os.path.join(prompt_dir, f"{args.target}_prompts_50_25.json")

    if not os.path.exists(prompt_file):
        print(f"⚠️ Prompt file not found: {prompt_file}")
        print(f"➡️ Using fallback auto-generated prompts instead.")
        prompts_data = {cls: [f"A photo of {cls}"] for cls in target_dataset.classnames}
    else:
        with open(prompt_file, "r") as f:
            prompts_data = json.load(f)

    all_prompts = []
    for cls_name in target_dataset.classnames:
        if cls_name in prompts_data:
            all_prompts.extend(prompts_data[cls_name])
        else:
            all_prompts.append(f"A photo of {cls_name}")


    # all_prompts = []
    # for cls_name in target_dataset.classnames:
    #     all_prompts.extend(prompts_data[cls_name])

    print("Encoding textual prompts...")
    with torch.no_grad():
        text_tokens = clip.tokenize(all_prompts).to(device)
        textual_prototypes = clip_model.encode_text(text_tokens)
        textual_prototypes = textual_prototypes / textual_prototypes.norm(dim=-1, keepdim=True)

    # Move tensors to CUDA
    v_prototypes = v_prototypes.to(device)
    v_labels = v_labels.to(device, dtype=torch.float16)
    textual_prototypes = textual_prototypes.to(device)

    print(f"Visual prototypes shape: {v_prototypes.shape}")
    print(f"v_labels shape: {v_labels.shape}")
    print(f"Textual prototypes shape: {textual_prototypes.shape}\n")

    # Adapters
    adapter = nn.Linear(v_prototypes.shape[0], v_prototypes.shape[1], bias=False)
    adapter.weight = nn.Parameter(v_prototypes.t())
    adapter = adapter.to(device, dtype=torch.float16)

    prompt_adapter = nn.Linear(textual_prototypes.shape[0], textual_prototypes.shape[1], bias=False)
    prompt_adapter.weight = nn.Parameter(textual_prototypes.t())
    prompt_adapter = prompt_adapter.to(device, dtype=torch.float16)

    # Domain adaptation module
    da_module = CORAL(lambda_coral=args.lambda_da)
    print(f"Using CORAL domain adaptation (λ = {args.lambda_da})\n")

    # Optimizers
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, eps=1e-4)
    prompt_optimizer = torch.optim.AdamW(prompt_adapter.parameters(), lr=args.lr, eps=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * min(len(source_train_loader), len(target_train_loader))
    )
    prompt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        prompt_optimizer, args.epochs * min(len(source_train_loader), len(target_train_loader))
    )

    # Extract test features (only once)
    print("Extracting test features...")
    test_features_list, test_labels_list = [], []
    for item in tqdm(target_dataset.test, desc="Test features"):
        img = Image.open(item.impath).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        test_features_list.append(feat.squeeze())
        test_labels_list.append(item.label)

    test_features = torch.stack(test_features_list).to(device)
    test_labels = torch.tensor(test_labels_list).to(device)

    best_acc = 0.0
    best_epoch = 0
    history = []

    print("\n" + "=" * 80)
    print("TRAINING LOOP START")
    print("=" * 80)

    for epoch in range(args.epochs):
        adapter.train()
        prompt_adapter.train()

        source_iter = iter(source_train_loader)
        target_iter = iter(target_train_loader)

        max_iter = min(len(source_train_loader), len(target_train_loader))

        epoch_loss = epoch_cls_loss = epoch_da_loss = 0.0
        correct, total = 0, 0

        pbar = tqdm(range(max_iter), desc=f"Epoch {epoch+1}/{args.epochs}")

        for _ in pbar:
            try:
                source_imgs, source_target = next(source_iter)
            except StopIteration:
                source_iter = iter(source_train_loader)
                source_imgs, source_target = next(source_iter)

            try:
                target_imgs, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_train_loader)
                target_imgs, _ = next(target_iter)

            source_imgs = source_imgs.to(device)
            source_target = source_target.to(device)
            target_imgs = target_imgs.to(device)

            with torch.no_grad():
                source_features = clip_model.encode_image(source_imgs)
                source_features = source_features / source_features.norm(dim=-1, keepdim=True)

                target_features = clip_model.encode_image(target_imgs)
                target_features = target_features / target_features.norm(dim=-1, keepdim=True)

            mvpdr_logits, v_logits, t_mean, t_max = compute_mvpdr_logits(
                source_features, adapter.weight.t(), prompt_adapter.weight.t(),
                v_labels, args.alpha, args.bbeta, args.gamma
            )

            cls_loss = (
                args.w1 * F.cross_entropy(v_logits, source_target) +
                args.w2 * F.cross_entropy(t_max, source_target) +
                args.w3 * F.cross_entropy(t_mean, source_target)
            )

            da_loss = da_module.compute_loss(
                source_features.float(), target_features.float()
            )

            total_loss = cls_loss + da_loss

            optimizer.zero_grad()
            prompt_optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(prompt_adapter.parameters(), 1.0)

            optimizer.step()
            prompt_optimizer.step()
            scheduler.step()
            prompt_scheduler.step()

            epoch_loss += total_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_da_loss += da_loss.item()

            pred = mvpdr_logits.argmax(dim=1)
            correct += (pred == source_target).sum().item()
            total += source_target.size(0)

            pbar.set_postfix({
                'loss': f'{total_loss.item():.3f}',
                'cls': f'{cls_loss.item():.3f}',
                'da': f'{da_loss.item():.3f}',
                'acc': f'{100*correct/total:.1f}%'
            })

        adapter.eval()
        prompt_adapter.eval()

        with torch.no_grad():
            test_logits, _, _, _ = compute_mvpdr_logits(
                test_features, adapter.weight.t(), prompt_adapter.weight.t(),
                v_labels, args.alpha, args.bbeta, args.gamma
            )
            pred = test_logits.argmax(dim=1)
            test_acc = (pred == test_labels).float().mean().item() * 100

        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Acc: {100*correct/total:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%")
        print(f"  Cls Loss: {epoch_cls_loss/max_iter:.4f}")
        print(f"  DA Loss: {epoch_da_loss/max_iter:.4f}")
        print(f"  Total Loss: {epoch_loss/max_iter:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_acc': 100*correct/total,
            'test_acc': test_acc,
            'cls_loss': epoch_cls_loss/max_iter,
            'da_loss': epoch_da_loss/max_iter,
            'total_loss': epoch_loss/max_iter
        })

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1

            save_dir = f'results_da/{args.target}/{args.backbone}/seed{args.seed}_lambda{args.lambda_da}'
            os.makedirs(save_dir, exist_ok=True)

            torch.save({
                'adapter_weight': adapter.weight.t(),
                'prompt_weight': prompt_adapter.weight.t(),
                'epoch': epoch + 1,
                'test_acc': test_acc
            }, os.path.join(save_dir, 'best_model_da.pth'))

            print(f"  🚀 New best model saved! (Acc: {test_acc:.2f}%)")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Best Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    print("=" * 80 + "\n")

    # Save history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='plantvillage')
    parser.add_argument('--target', default='plantdoc')
    parser.add_argument('--data_root', default='')
    parser.add_argument('--backbone', default='RN101')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--bbeta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--w1', type=float, default=1.0)
    parser.add_argument('--w2', type=float, default=0.5)
    parser.add_argument('--w3', type=float, default=0.5)
    parser.add_argument('--lambda_da', type=float, default=0.5)

    args = parser.parse_args()
    train_with_domain_adaptation(args)
