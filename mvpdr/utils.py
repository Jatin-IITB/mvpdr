import json
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.cluster import KMeans
from tqdm import tqdm

from mvpdr import clip


def cls_acc(output, target, topk=1, labels=None):
    pred = output.topk(topk, 1, True, True)[1].t()
    y_pred = pred.squeeze().tolist()
    y_true = target.tolist()

    if isinstance(y_pred, int):
        y_pred = [y_pred]
    if isinstance(y_true, (int, float)):
        y_true = [y_true]

    return {
        "acc": metrics.accuracy_score(y_true, y_pred) * 100,
        "precision": metrics.precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0) * 100,
        "recall": metrics.recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0) * 100,
        "f1": metrics.f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0) * 100,
    }


def cls_acc_test(output, target, topk=1, labels=None):
    result = cls_acc(output, target, topk=topk, labels=labels)

    pred = output.topk(topk, 1, True, True)[1].squeeze()
    n_classes = int(torch.max(target).item()) + 1
    conf_matrix = torch.zeros(n_classes, n_classes, device=pred.device)
    for p, t in zip(pred, target):
        conf_matrix[int(p.item()), int(t.item())] += 1

    result["conf_matrix"] = conf_matrix.cpu()
    return result


def clip_classifier(classnames, template, clip_model):
    device = next(clip_model.parameters()).device
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace("_", " ")
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights


def build_textual_prototypes(classnames, template, clip_model, path):
    device = next(clip_model.parameters()).device
    with torch.no_grad():
        with open(path, "r") as f:
            json_data = json.load(f)

        clip_weights = []
        for classname in classnames:
            texts = json_data[classname]
            texts = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            clip_weights.append(class_embeddings)

        clip_weights = torch.cat(clip_weights, dim=0).to(device).t()
    return clip_weights


def build_visual_prototypes(cfg, clip_model, train_loader_cache, n_cls, n_clt):
    device = next(clip_model.parameters()).device

    if not cfg["load_cache"]:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            for augment_idx in range(cfg["augment_epoch"]):
                train_features = []
                print(f"Augment Epoch: {augment_idx} / {cfg['augment_epoch']}")
                for images, target in tqdm(train_loader_cache):
                    images = images.to(device)
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        cache_values.append(target.to(device))
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_values = torch.cat(cache_values, dim=0)

        feature_dict = {l: [] for l in range(n_cls)}
        for i in range(len(cache_values)):
            feature_dict[cache_values[i].item()].append(cache_keys[i].unsqueeze(0))

        features_list = []
        label_list = []
        min_samples = float("inf")

        print("\nBuilding prototypes per class...")
        for k in feature_dict:
            tensors = torch.cat(feature_dict[k], dim=0)
            n_samples = tensors.shape[0]
            min_samples = min(min_samples, n_samples)
            effective_n_clt = min(n_clt, max(1, n_samples))

            if n_samples == 1:
                item = tensors
            elif effective_n_clt == 1:
                item = tensors.mean(dim=0, keepdim=True)
            else:
                im_arr = tensors.detach().cpu().numpy()
                kmeans = KMeans(n_clusters=effective_n_clt, random_state=0, n_init=10)
                kmeans.fit(im_arr)
                item = torch.from_numpy(kmeans.cluster_centers_).to(tensors.dtype)

            if effective_n_clt < n_clt:
                warnings.warn(f"Class {k}: {n_samples} samples, using {effective_n_clt} prototypes instead of {n_clt}")

            features_list.append(item)
            label_list += [k] * item.shape[0]

        cache_keys = torch.cat(features_list).permute(1, 0).half()
        cache_values = F.one_hot(torch.tensor(label_list, dtype=torch.int64)).half()

        print(f"Created {cache_keys.shape[1]} visual prototypes across {n_cls} classes")

        torch.save(cache_keys, f"{cfg['cache_dir']}/keys_{n_clt}clts.pt")
        torch.save(cache_values, f"{cfg['cache_dir']}/values_{n_clt}clts.pt")
    else:
        cache_keys = torch.load(f"{cfg['cache_dir']}/keys_{n_clt}clts.pt")
        cache_values = torch.load(f"{cfg['cache_dir']}/values_{n_clt}clts.pt")

    return cache_keys.to(device), cache_values.to(device)


def pre_load_features(cfg, split, clip_model, loader):
    device = next(clip_model.parameters()).device

    if not cfg["load_pre_feat"]:
        features, labels = [], []
        with torch.no_grad():
            for images, target in tqdm(loader):
                images, target = images.to(device), target.to(device)
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)
        torch.save(features, f"{cfg['cache_dir']}/{split}_f.pt")
        torch.save(labels, f"{cfg['cache_dir']}/{split}_l.pt")
    else:
        features = torch.load(f"{cfg['cache_dir']}/{split}_f.pt")
        labels = torch.load(f"{cfg['cache_dir']}/{split}_l.pt")

    return features.half(), labels.long()
