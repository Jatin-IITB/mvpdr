import os
import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import transforms
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
import warnings
import json
import matplotlib
matplotlib.use('Agg')  # For server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

warnings.filterwarnings("ignore")

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_arguments():
    parser = argparse.ArgumentParser(description='MVPDR - Multi-View Prototype-based Disease Recognition')
    parser.add_argument('--config', dest='config', default="configs/plantdoc_clt.yaml", help='Path to config file')
    parser.add_argument('--nclt', type=int, default=16, help='Number of clusters for visual prototypes')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--backbone', type=str, default="RN101", choices=['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'], help='CLIP backbone')
    parser.add_argument('--w1', type=float, default=1, help='Weight for visual prototype loss')
    parser.add_argument('--w2', type=float, default=0.1, help='Weight for textual max loss')
    parser.add_argument('--w3', type=float, default=0.1, help='Weight for textual mean loss')
    parser.add_argument('--alpha', type=float, default=0.3, help='Visual-textual fusion weight')
    parser.add_argument('--bbeta', type=float, default=0.5, help='Beta parameter for prototype affinity')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma parameter for textual fusion')
    parser.add_argument('--gpu', type=str, default="0", help='GPU device ID')
    args = parser.parse_args()
    return args


def save_training_curves(train_history, save_path):
    """Save training loss and accuracy curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Loss curve
    axes[0, 0].plot(train_history['epoch'], train_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Training accuracy
    axes[0, 1].plot(train_history['epoch'], train_history['train_acc'], 'g-', label='Training Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training Accuracy Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Test metrics over epochs
    axes[1, 0].plot(train_history['epoch'], train_history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
    axes[1, 0].plot(train_history['epoch'], train_history['test_f1'], 'b--', label='Test F1-Score', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Score (%)', fontsize=12)
    axes[1, 0].set_title('Test Performance Over Epochs', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[1, 1].plot(train_history['epoch'], train_history['learning_rate'], 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def save_confusion_matrix(conf_matrix, class_names, save_path):
    """Save confusion matrix visualization"""
    plt.figure(figsize=(max(12, len(class_names) * 0.4), max(10, len(class_names) * 0.35)))

    # Normalize confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    sns.heatmap(conf_matrix_norm, annot=False, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})

    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def save_per_class_metrics(class_report_dict, save_path):
    """Save per-class performance metrics"""
    # Extract per-class metrics
    classes = [k for k in class_report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]

    if not classes:
        print("Warning: No per-class metrics found")
        return

    precision = [class_report_dict[c]['precision'] * 100 for c in classes]
    recall = [class_report_dict[c]['recall'] * 100 for c in classes]
    f1_score = [class_report_dict[c]['f1-score'] * 100 for c in classes]
    support = [class_report_dict[c]['support'] for c in classes]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    x_pos = np.arange(len(classes))

    # Precision by class
    axes[0, 0].bar(x_pos, precision, color='skyblue', alpha=0.8)
    axes[0, 0].set_ylabel('Precision (%)', fontsize=12)
    axes[0, 0].set_title('Precision by Class', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(y=np.mean(precision), color='r', linestyle='--', label=f'Mean: {np.mean(precision):.2f}%')
    axes[0, 0].legend()

    # Recall by class
    axes[0, 1].bar(x_pos, recall, color='lightcoral', alpha=0.8)
    axes[0, 1].set_ylabel('Recall (%)', fontsize=12)
    axes[0, 1].set_title('Recall by Class', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].axhline(y=np.mean(recall), color='r', linestyle='--', label=f'Mean: {np.mean(recall):.2f}%')
    axes[0, 1].legend()

    # F1-Score by class
    axes[1, 0].bar(x_pos, f1_score, color='lightgreen', alpha=0.8)
    axes[1, 0].set_ylabel('F1-Score (%)', fontsize=12)
    axes[1, 0].set_title('F1-Score by Class', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].axhline(y=np.mean(f1_score), color='r', linestyle='--', label=f'Mean: {np.mean(f1_score):.2f}%')
    axes[1, 0].legend()

    # Support by class
    axes[1, 1].bar(x_pos, support, color='plum', alpha=0.8)
    axes[1, 1].set_ylabel('Number of Samples', fontsize=12)
    axes[1, 1].set_title('Test Samples per Class', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics saved to {save_path}")


def run_MVPDR(cfg, v_prototypes, v_labels, test_features, test_labels, textual_prototypes,
              clip_model, train_loader_F, weights, class_names):
    """Main training loop with comprehensive logging"""

    n_class = v_labels.shape[-1]

    # Initialize adapters
    adapter = nn.Linear(v_prototypes.shape[0], v_prototypes.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(v_prototypes.t())

    prompt_adapter = nn.Linear(textual_prototypes.shape[0], textual_prototypes.shape[1], bias=False).to(clip_model.dtype).cuda()
    prompt_adapter.weight = nn.Parameter(textual_prototypes.t())

    # Optimizers
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    prompt_optimizer = torch.optim.AdamW(prompt_adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    prompt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(prompt_optimizer, cfg['train_epoch'] * len(train_loader_F))

    best_acc, best_epoch = 0.0, 0
    gamma, bbeta, alpha = cfg['gamma'], cfg['bbeta'], cfg['alpha']
    labels = np.unique(list(range(n_class)))

    # Training history
    train_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'test_precision': [],
        'test_recall': [],
        'test_f1': [],
        'learning_rate': []
    }

    print("\n" + "="*80)
    print("Starting MVPDR Training")
    print("="*80)

    for train_idx in range(cfg['train_epoch']):
        adapter.train()
        prompt_adapter.train()

        correct_samples, all_samples = 0, 0
        loss_list = []

        print(f'\nEpoch [{train_idx + 1}/{cfg["train_epoch"]}]')

        for i, (images, target) in enumerate(tqdm(train_loader_F, desc=f"Training")):
            images, target = images.cuda(), target.cuda()

            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # Forward pass
            affinity = adapter(image_features)
            v_logits = ((-1) * (bbeta - bbeta * affinity)).exp() @ v_labels

            t_logits = 100. * prompt_adapter(image_features)
            t_logits = t_logits.reshape(t_logits.shape[0], n_class, -1)
            t_mean_logits = t_logits.mean(dim=-1)
            t_max_logits = t_logits.max(dim=-1)[0]
            t_logits = gamma * t_mean_logits + bbeta * t_max_logits

            MVPDR_logits = t_logits + v_logits * alpha

            # Loss computation
            w1, w2, w3 = weights
            loss1 = F.cross_entropy(v_logits, target)
            loss3 = F.cross_entropy(t_max_logits, target)
            loss4 = F.cross_entropy(t_mean_logits, target)
            loss = w1 * loss1 + w2 * loss3 + w3 * loss4

            acc = cls_acc(MVPDR_logits, target, labels=labels)["acc"]
            correct_samples += acc / 100 * len(MVPDR_logits)
            all_samples += len(MVPDR_logits)
            loss_list.append(loss.item())

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            prompt_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(prompt_adapter.parameters(), max_norm=1.0)
            optimizer.step()
            prompt_optimizer.step()
            scheduler.step()
            prompt_scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        train_acc = correct_samples / all_samples
        train_loss = sum(loss_list) / len(loss_list)

        print(f'  LR: {current_lr:.6f}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}')

        # Evaluation on test set
        adapter.eval()
        prompt_adapter.eval()

        with torch.no_grad():
            affinity = adapter(test_features)
            v_logits = ((-1) * (bbeta - bbeta * affinity)).exp() @ v_labels

            t_logits = 100. * prompt_adapter(test_features)
            t_logits = t_logits.reshape(t_logits.shape[0], n_class, -1)
            t_mean_logits = t_logits.mean(dim=-1)
            t_max_logits = t_logits.max(dim=-1)[0]
            t_logits = gamma * t_mean_logits + bbeta * t_max_logits

            MVPDR_logits = t_logits + v_logits * alpha

        result = cls_acc(MVPDR_logits, test_labels, labels=labels)
        acc, precision, recall, f1_score = result["acc"], result["precision"], result["recall"], result["f1"]

        print(f'  Test Acc: {acc:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1: {f1_score:.2f}%')

        # Save history
        train_history['epoch'].append(train_idx + 1)
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc * 100)
        train_history['test_acc'].append(acc)
        train_history['test_precision'].append(precision)
        train_history['test_recall'].append(recall)
        train_history['test_f1'].append(f1_score)
        train_history['learning_rate'].append(current_lr)

        # Save best model
        if acc > best_acc:
            best_acc, best_precision, best_recall, best_f1, best_epoch = acc, precision, recall, f1_score, train_idx + 1
            torch.save(adapter.weight, os.path.join(cfg['cache_dir'], f"best_F_{cfg['shots']}shots.pt"))
            torch.save(prompt_adapter.weight, os.path.join(cfg['cache_dir'], "best_prompt.pt"))
            print(f'  *** New best model saved! ***')

    print("\n" + "="*80)
    print(f"Training completed! Best accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    print("="*80 + "\n")

    # Load best model for final evaluation
    adapter.weight = torch.load(os.path.join(cfg['cache_dir'], f"best_F_{cfg['shots']}shots.pt"))
    prompt_adapter.weight = torch.load(os.path.join(cfg['cache_dir'], "best_prompt.pt"))

    adapter.eval()
    prompt_adapter.eval()

    # Final evaluation with recomputed logits
    with torch.no_grad():
        affinity = adapter(test_features)
        v_logits = ((-1) * (bbeta - bbeta * affinity)).exp() @ v_labels

        t_logits = 100. * prompt_adapter(test_features)
        t_logits = t_logits.reshape(t_logits.shape[0], n_class, -1)
        t_mean_logits = t_logits.mean(dim=-1)
        t_max_logits = t_logits.max(dim=-1)[0]
        t_logits = gamma * t_mean_logits + bbeta * t_max_logits

        MVPDR_logits = t_logits + v_logits * alpha

    result = cls_acc_test(MVPDR_logits, test_labels, labels=labels)
    conf_matrix = np.array(result["conf_matrix"].cpu())

    # Get detailed predictions
    pred_labels = MVPDR_logits.topk(1, 1, True, True)[1].squeeze().cpu().numpy()
    true_labels = test_labels.cpu().numpy()

    # Generate classification report
    class_report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True, zero_division=0)

    return {
        'best_acc': best_acc,
        'best_precision': best_precision,
        'best_recall': best_recall,
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'train_history': train_history,
        'conf_matrix': conf_matrix,
        'class_report': class_report,
        'predictions': pred_labels,
        'true_labels': true_labels
    }


def main():
    start_time = time.time()

    # Parse arguments
    args = get_arguments()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config
    assert os.path.exists(args.config), f"Config file not found: {args.config}"
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # Setup directories
    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    # Update config with args
    cfg['backbone'] = args.backbone
    cfg['init_alpha'] = args.alpha
    cfg["weights"] = [args.w1, args.w2, args.w3]
    cfg["alpha"], cfg["bbeta"], cfg["gamma"] = args.alpha, args.bbeta, args.gamma

    # Create output directory
    output_dir = f"results/{cfg['dataset']}/{args.backbone}/seed{args.seed}_alpha{args.alpha:.1f}_nclt{args.nclt}"
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed
    set_random_seed(args.seed)

    print("\n" + "="*80)
    print("MVPDR Configuration")
    print("="*80)
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    print("="*80 + "\n")

    # Load CLIP
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()
    preprocess_size = preprocess.__dict__['transforms'][0].size

    # Dataset preparation
    print("Preparing dataset...")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    val_loader = build_data_loader(dataset.val, batch_size=32, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(dataset.test, batch_size=32, is_train=False, tfm=preprocess, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=preprocess_size, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = build_data_loader(dataset.train_x, batch_size=32, tfm=train_tranform, is_train=True, shuffle=False)
    train_loader_F = build_data_loader(dataset.train_x, batch_size=32, tfm=train_tranform, is_train=True, shuffle=True)

    # Build textual prototypes
    print("Building textual prototypes...")
    path_dict = {
        "plantwild": "gpt_files_plt44/plantwild_prompts_50_18.json",
        "plantdoc": "gpt_files_plt44/plantdoc_prompts_50_25.json",
        "plantvillage": "gpt_files_plt44/plantvillage_prompts_50_25.json"
    }

    if cfg['dataset'] in path_dict:
        textual_prototypes = build_textual_prototypes(
            dataset.classnames if cfg['dataset'] != "plantvillage" else dataset.origin_classes,
            dataset.template, clip_model, path_dict[cfg['dataset']]
        )
    else:
        textual_prototypes = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Build visual prototypes
    v_prototypes_path = os.path.join(cfg['cache_dir'], f"v_prototypes_{cfg['shots']}shots.pt")
    v_labels_path = os.path.join(cfg['cache_dir'], f"v_labels_{cfg['shots']}shots.pt")

    if cfg['load_cache'] and os.path.exists(v_prototypes_path) and os.path.exists(v_labels_path):
        print("Loading cached visual prototypes...")
        v_prototypes = torch.load(v_prototypes_path)
        v_labels = torch.load(v_labels_path)
    else:
        print("Constructing visual prototypes...")
        v_prototypes, v_labels = build_visual_prototypes(cfg, clip_model, train_loader_cache, len(dataset.classnames), n_clt=args.nclt)
        torch.save(v_prototypes, v_prototypes_path)
        torch.save(v_labels, v_labels_path)

    # Precompute test features
    test_feat_path = os.path.join(cfg['cache_dir'], "test_features.pt")
    test_labels_path = os.path.join(cfg['cache_dir'], "test_labels.pt")

    if cfg['load_pre_feat'] and os.path.exists(test_feat_path) and os.path.exists(test_labels_path):
        print("Loading cached test features...")
        test_features = torch.load(test_feat_path)
        test_labels = torch.load(test_labels_path)
    else:
        print("Extracting test features...")
        test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
        torch.save(test_features, test_feat_path)
        torch.save(test_labels, test_labels_path)

    # Run MVPDR training
    results = run_MVPDR(
        cfg, v_prototypes, v_labels, test_features, test_labels,
        textual_prototypes, clip_model, train_loader_F, cfg["weights"],
        dataset.classnames
    )

    elapsed_time = time.time() - start_time

    # Save comprehensive results
    print("\nSaving results...")

    # 1. Save training curves
    save_training_curves(results['train_history'], os.path.join(output_dir, 'training_curves.png'))

    # 2. Save confusion matrix
    save_confusion_matrix(results['conf_matrix'], dataset.classnames, os.path.join(output_dir, 'confusion_matrix.png'))

    # 3. Save per-class metrics
    save_per_class_metrics(results['class_report'], os.path.join(output_dir, 'per_class_metrics.png'))

    # 4. Save detailed results to JSON
    results_json = {
        'dataset': cfg['dataset'],
        'backbone': args.backbone,
        'seed': args.seed,
        'hyperparameters': {
            'nclt': args.nclt,
            'w1': args.w1,
            'w2': args.w2,
            'w3': args.w3,
            'alpha': args.alpha,
            'bbeta': args.bbeta,
            'gamma': args.gamma,
            'learning_rate': cfg['lr'],
            'train_epochs': cfg['train_epoch'],
            'shots': cfg['shots']
        },
        'performance': {
            'best_accuracy': float(results['best_acc']),
            'best_precision': float(results['best_precision']),
            'best_recall': float(results['best_recall']),
            'best_f1_score': float(results['best_f1']),
            'best_epoch': int(results['best_epoch'])
        },
        'training_time_seconds': float(elapsed_time),
        'per_class_metrics': results['class_report']
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=4)

    # 5. Save training history to CSV
    history_df = pd.DataFrame(results['train_history'])
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

    # 6. Save detailed classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write("MVPDR Classification Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {cfg['dataset']}\n")
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write(f"Training Time: {elapsed_time/60:.2f} minutes\n\n")
        f.write("="*80 + "\n")
        f.write("Overall Performance\n")
        f.write("="*80 + "\n")
        f.write(f"Best Test Accuracy:  {results['best_acc']:.2f}%\n")
        f.write(f"Best Precision:      {results['best_precision']:.2f}%\n")
        f.write(f"Best Recall:         {results['best_recall']:.2f}%\n")
        f.write(f"Best F1-Score:       {results['best_f1']:.2f}%\n")
        f.write(f"Best Epoch:          {results['best_epoch']}/{cfg['train_epoch']}\n\n")

        f.write("="*80 + "\n")
        f.write("Per-Class Metrics\n")
        f.write("="*80 + "\n\n")

        # Write detailed class-wise performance
        f.write(f"{'Class':<40} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-"*90 + "\n")

        for cls_name in dataset.classnames:
            if cls_name in results['class_report']:
                metrics = results['class_report'][cls_name]
                f.write(f"{cls_name:<40} {metrics['precision']*100:>10.2f}% {metrics['recall']*100:>10.2f}% "
                       f"{metrics['f1-score']*100:>10.2f}% {int(metrics['support']):>10}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Macro Averages\n")
        f.write("="*80 + "\n")
        f.write(f"Precision: {results['class_report']['macro avg']['precision']*100:.2f}%\n")
        f.write(f"Recall:    {results['class_report']['macro avg']['recall']*100:.2f}%\n")
        f.write(f"F1-Score:  {results['class_report']['macro avg']['f1-score']*100:.2f}%\n")

    # 7. Save model weights
    full_model_path = os.path.join(output_dir, f"mvpdr_model.pth")
    torch.save({
        'adapter_weight': torch.load(os.path.join(cfg['cache_dir'], f"best_F_{cfg['shots']}shots.pt")),
        'prompt_weight': torch.load(os.path.join(cfg['cache_dir'], "best_prompt.pt")),
        'clip_backbone': cfg['backbone'],
        'config': cfg,
        'performance': results_json['performance']
    }, full_model_path)

    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    print(f"Test Accuracy:  {results['best_acc']:.2f}%")
    print(f"Precision:      {results['best_precision']:.2f}%")
    print(f"Recall:         {results['best_recall']:.2f}%")
    print(f"F1-Score:       {results['best_f1']:.2f}%")
    print(f"Best Epoch:     {results['best_epoch']}/{cfg['train_epoch']}")
    print(f"Training Time:  {elapsed_time/60:.2f} minutes")
    print(f"\nAll results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
