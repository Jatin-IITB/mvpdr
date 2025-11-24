import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from datasets import build_dataset
import clip
from openset_detector import OpenSetDetector

def load_cache_files(cache_dir, shots=0):
    """Load prototypes and labels from cache"""
    v_prototypes_path = os.path.join(cache_dir, f'v_prototypes_{shots}shots.pt')
    v_labels_path = os.path.join(cache_dir, f'v_labels_{shots}shots.pt')

    if not os.path.exists(v_prototypes_path):
        raise FileNotFoundError(f"Prototypes not found: {v_prototypes_path}")
    if not os.path.exists(v_labels_path):
        raise FileNotFoundError(f"Labels not found: {v_labels_path}")

    v_prototypes = torch.load(v_prototypes_path)
    v_labels = torch.load(v_labels_path)

    print(f"✅ Loaded from cache:")
    print(f"   Prototypes: {v_prototypes.shape}")
    print(f"   Labels: {v_labels.shape}")

    return v_prototypes, v_labels

def load_trained_weights(results_dir, dataset, backbone, seed, alpha, nclt):
    """Load trained adapter weights"""
    model_dir = os.path.join(results_dir, dataset, backbone, f'seed{seed}_alpha{alpha}_nclt{nclt}')
    checkpoint_path = os.path.join(model_dir, 'mvpdr_model.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    adapter_weight = checkpoint['adapter_weight'].t()  # Transpose for correct dims
    prompt_weight = checkpoint['prompt_weight'].t()

    print(f"✅ Loaded trained weights:")
    print(f"   Visual adapter: {adapter_weight.shape} ({adapter_weight.dtype})")
    print(f"   Textual adapter: {prompt_weight.shape} ({prompt_weight.dtype})")

    return adapter_weight, prompt_weight

def compute_mvpdr_logits(image_features, adapter_weight, prompt_weight, v_labels, 
                          alpha=0.3, bbeta=0.5, gamma=0.5):
    """Compute MVPDR logits"""
    batch_size = image_features.shape[0]
    num_classes = v_labels.shape[1]

    # Visual path
    affinity = image_features @ adapter_weight
    v_logits = ((-1) * (bbeta - bbeta * affinity)).exp() @ v_labels

    # Textual path
    t_logits = 100. * (image_features @ prompt_weight)
    t_logits = t_logits.reshape(batch_size, num_classes, -1)
    t_mean_logits = t_logits.mean(dim=-1)
    t_max_logits = t_logits.max(dim=-1)[0]
    t_logits = gamma * t_mean_logits + bbeta * t_max_logits

    # Final
    return t_logits + v_logits * alpha

def extract_test_logits(clip_model, preprocess, test_split, adapter_weight, 
                         prompt_weight, v_labels, cfg, device):
    """Extract logits from test set"""
    all_logits = []
    all_labels = []

    clip_model.eval()
    batch_size = 32

    for i in tqdm(range(0, len(test_split), batch_size), desc="Extracting logits"):
        batch = test_split[i:i+batch_size]

        images = []
        labels = []

        for item in batch:
            img = Image.open(item.impath).convert('RGB')
            images.append(preprocess(img))
            labels.append(item.label)

        images = torch.stack(images).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = compute_mvpdr_logits(
                image_features, adapter_weight, prompt_weight, v_labels,
                cfg['alpha'], cfg['bbeta'], cfg['gamma']
            )

        all_logits.append(logits.cpu())
        all_labels.extend(labels)

    return torch.cat(all_logits, 0), torch.tensor(all_labels)

def main(args):
    print("\n" + "="*80)
    print("MVPDR OPEN-SET DETECTION - CLEAN IMPLEMENTATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.method}")
    print(f"Unknown classes: {args.unknown_classes}")
    print("="*80 + "\n")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP
    clip_model, preprocess = clip.load(args.backbone, device)
    print(f"✅ Loaded CLIP {args.backbone}\n")

    # Load dataset
    dataset = build_dataset(args.dataset, args.data_root, shots=0)
    num_classes = len(dataset.classnames)
    print(f"Dataset: {num_classes} classes\n")

    # Load cache files
    cache_dir = os.path.join('caches', args.dataset) if not args.cache_dir else args.cache_dir
    v_prototypes, v_labels = load_cache_files(cache_dir, shots=0)
    v_labels = v_labels.to(device, dtype=torch.float16)
    print()

    # Load trained weights
    adapter_weight, prompt_weight = load_trained_weights(
        args.results_dir, args.dataset, args.backbone, 
        args.seed, args.alpha, args.nclt
    )
    adapter_weight = adapter_weight.to(device)
    prompt_weight = prompt_weight.to(device)
    print()

    # Split known/unknown classes
    all_classes = list(range(num_classes))
    unknown_class_ids = sorted(np.random.choice(all_classes, args.unknown_classes, replace=False))
    known_class_ids = [c for c in all_classes if c not in unknown_class_ids]

    print(f"Known classes ({len(known_class_ids)}): {known_class_ids}")
    print(f"Unknown classes ({len(unknown_class_ids)}): {unknown_class_ids}")
    print(f"Unknown: {[dataset.classnames[i] for i in unknown_class_ids]}")
    print()

    # Extract logits
    cfg = {'alpha': args.alpha, 'bbeta': args.bbeta, 'gamma': args.gamma}
    print("Extracting MVPDR logits...")
    all_logits, all_labels = extract_test_logits(
        clip_model, preprocess, dataset.test, adapter_weight,
        prompt_weight, v_labels, cfg, device
    )

    # Split known/unknown
    known_mask = torch.tensor([label.item() in known_class_ids for label in all_labels])
    known_logits = all_logits[known_mask]
    unknown_logits = all_logits[~known_mask]

    print(f"\nKnown samples: {len(known_logits)}, Unknown samples: {len(unknown_logits)}")

    # Evaluate
    detector = OpenSetDetector(method=args.method)
    results = detector.evaluate(known_logits, unknown_logits)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"AUROC: {results['auroc']:.4f}")
    print(f"AUPR: {results['aupr']:.4f}")
    print(f"FPR@95%% TPR: {results['fpr_at_95tpr']:.4f}")
    print(f"Detection Acc: {results['detection_accuracy']*100:.2f}%%")
    print("="*80 + "\n")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    detector.save_results(results, args.output_dir)

    # Save config
    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
        f.write(f"MVPDR Open-Set Detection\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"AUROC: {results['auroc']:.4f}\n")
        f.write(f"Unknown: {[dataset.classnames[i] for i in unknown_class_ids]}\n")

    print(f"✅ Results saved to: {args.output_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='plantdoc')
    parser.add_argument('--data_root', default='')
    parser.add_argument('--cache_dir', default='')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--backbone', default='RN101')
    parser.add_argument('--method', default='msp', choices=['msp', 'energy'])
    parser.add_argument('--unknown_classes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--bbeta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--nclt', type=int, default=16)
    parser.add_argument('--output_dir', default='openset_results/msp')

    args = parser.parse_args()
    main(args)
