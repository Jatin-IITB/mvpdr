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
warnings.filterwarnings("ignore")

random.seed(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default="configs/plantdoc_clt.yaml")
    parser.add_argument('--n_clt', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--backbone', type=str, default="RN101")
    parser.add_argument('--w1', type=float, default=1)
    parser.add_argument('--w2', type=float, default=0.1)
    parser.add_argument('--w3', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--bbeta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.5)
    args = parser.parse_args()
    return args

def run_MVPDR(cfg, v_prototypes, v_labels, test_features, test_labels, textual_prototypes,
             clip_model, train_loader_F, weights):
    
    n_class = v_labels.shape[-1]
    adapter = nn.Linear(v_prototypes.shape[0], v_prototypes.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(v_prototypes.t())

    prompt_adapter = nn.Linear(textual_prototypes.shape[0], textual_prototypes.shape[1], bias=False).to(clip_model.dtype).cuda()
    prompt_adapter.weight = nn.Parameter(textual_prototypes.t())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    prompt_optimizer = torch.optim.AdamW(prompt_adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    prompt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(prompt_optimizer, cfg['train_epoch'] * len(train_loader_F))

    best_acc, best_epoch = 0.0, 0
    gamma, bbeta, alpha = cfg['gamma'], cfg['bbeta'], cfg['alpha']
    labels = np.unique(list(range(n_class)))

    for train_idx in range(cfg['train_epoch']):
        adapter.train()
        prompt_adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []

        print(f'Train Epoch: {train_idx} / {cfg["train_epoch"]}')
        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            v_logits = ((-1) * (bbeta - bbeta * affinity)).exp() @ v_labels
            t_logits = 100. * prompt_adapter(image_features)
            t_logits = t_logits.reshape(t_logits.shape[0], n_class, -1)
            t_mean_logits = t_logits.mean(dim=-1)
            t_max_logits = t_logits.max(dim=-1)[0]
            t_logits = gamma * t_mean_logits + bbeta * t_max_logits

            MVPDR_logits = t_logits + v_logits * alpha
            w1, w2, w3 = weights

            loss1 = F.cross_entropy(v_logits, target)
            loss3 = F.cross_entropy(t_max_logits, target)
            loss4 = F.cross_entropy(t_mean_logits, target)
            loss = w1 * loss1 + w2 * loss3 + w3 * loss4

            acc = cls_acc(MVPDR_logits, target, labels=labels)["acc"]
            correct_samples += acc / 100 * len(MVPDR_logits)
            all_samples += len(MVPDR_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            prompt_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prompt_optimizer.step()
            scheduler.step()
            prompt_scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f'LR: {current_lr:.6f}, Acc: {correct_samples/all_samples:.4f} ({correct_samples}/{all_samples}), Loss: {sum(loss_list)/len(loss_list):.4f}')

        # Eval on test
        adapter.eval()
        prompt_adapter.eval()

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

        print(f"**** MVPDR test accuracy: {acc:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1: {f1_score:.2f}. ****\n")
        if acc > best_acc:
            best_acc, best_precision, best_recall, best_f1, best_epoch = acc, precision, recall, f1_score
            torch.save(adapter.weight, os.path.join(cfg['cache_dir'], f"best_F_{cfg['shots']}shots.pt"))
            torch.save(prompt_adapter.weight, os.path.join(cfg['cache_dir'], "best_prompt.pt"))

    adapter.weight = torch.load(os.path.join(cfg['cache_dir'], f"best_F_{cfg['shots']}shots.pt"))
    print(f"**** After fine-tuning, MVPDR best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    # Final evaluation
    affinity = adapter(test_features)
    v_logits = ((-1) * (bbeta - bbeta * affinity)).exp() @ v_labels
    MVPDR_logits = t_logits + v_logits * alpha
    result = cls_acc_test(MVPDR_logits, test_labels, labels=labels)
    conf_matrix = np.array(result["conf_matrix"].cpu())

    print(f"**** MVPDR test accuracy: {result['acc']:.2f}, precision: {result['precision']:.3f}, recall: {result['recall']:.3f}, f1: {result['f1']:.3f}. ****\n")

    return best_acc, best_precision, best_recall, best_f1, best_epoch

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    start_time = time.time()
    args = get_arguments()
    assert os.path.exists(args.config)
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    cfg['backbone'] = args.backbone
    cfg['init_alpha'] = args.alpha
    cfg["weights"] = [args.w1, args.w2, args.w3]
    cfg["alpha"], cfg["bbeta"], cfg["gamma"] = args.alpha, args.bbeta, args.gamma

    print("\nRunning configs.")
    print(cfg, "\n")

    # Load CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()
    preprocess_size = preprocess.__dict__['transforms'][0].size

    # Dataset preparation
    set_random_seed(args.seed)
    print("Preparing dataset.")
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

    # Textual features
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

    # --- Visual prototypes caching ---
    v_prototypes_path = os.path.join(cfg['cache_dir'], f"v_prototypes_{cfg['shots']}shots.pt")
    v_labels_path = os.path.join(cfg['cache_dir'], f"v_labels_{cfg['shots']}shots.pt")
    if cfg['load_cache'] and os.path.exists(v_prototypes_path) and os.path.exists(v_labels_path):
        print("Loading cached visual prototypes...")
        v_prototypes = torch.load(v_prototypes_path)
        v_labels = torch.load(v_labels_path)
    else:
        print("Constructing visual prototypes...")
        v_prototypes, v_labels = build_visual_prototypes(cfg, clip_model, train_loader_cache, len(dataset.classnames), n_clt=args.n_clt)
        torch.save(v_prototypes, v_prototypes_path)
        torch.save(v_labels, v_labels_path)

    # --- Precomputed CLIP features caching ---
    test_feat_path = os.path.join(cfg['cache_dir'], "test_features.pt")
    test_labels_path = os.path.join(cfg['cache_dir'], "test_labels.pt")
    if cfg['load_pre_feat'] and os.path.exists(test_feat_path) and os.path.exists(test_labels_path):
        print("Loading cached test features...")
        test_features = torch.load(test_feat_path)
        test_labels = torch.load(test_labels_path)
    else:
        test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
        torch.save(test_features, test_feat_path)
        torch.save(test_labels, test_labels_path)

    # --- Run MVPDR ---
    best_acc, best_precision, best_recall, best_f1_score, best_epoch = run_MVPDR(
        cfg, v_prototypes, v_labels, test_features, test_labels,
        textual_prototypes, clip_model, train_loader_F, cfg["weights"]
    )

    # Save full model weights
    full_model_path = f"mvpdr_{cfg['dataset']}_full.pth"
    torch.save({
        'adapter_weight': torch.load(os.path.join(cfg['cache_dir'], f"best_F_{cfg['shots']}shots.pt")),
        'prompt_weight': torch.load(os.path.join(cfg['cache_dir'], "best_prompt.pt")),
        'clip_backbone': cfg['backbone']
    }, full_model_path)
    print(f"Saved full MVPDR model to {full_model_path}")

    elapsed_time = time.time() - start_time
    output_path = f"outputs/{cfg['dataset']}/{args.backbone}/seed_{args.seed}"
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(
        output_path,
        f"{cfg['dataset']}_{args.w1}_{args.w2}_{args.w3}_{cfg['alpha']}_{cfg['bbeta']}_{cfg['gamma']}.txt"
    )
    with open(output_file, "w") as f:
        f.write(f"**** MVPDR test accuracy: {best_acc:.2f}, precision: {best_precision:.3f}, recall: {best_recall:.3f}, f1: {best_f1_score:.3f}. ****\n")
        f.write(f"Best epoch: {best_epoch}/{cfg['train_epoch']}\n")
        f.write(f"Time used: {elapsed_time}")

if __name__ == '__main__':
    main()
