import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from mvpdr import clip


def load_model(model_path, device="cuda"):
    checkpoint = torch.load(model_path, map_location=device)
    clip_model, preprocess = clip.load(checkpoint["clip_backbone"], device=device)
    clip_model.eval()

    adapter = nn.Linear(checkpoint["adapter_weight"].shape[1], checkpoint["adapter_weight"].shape[0], bias=False)
    adapter.weight = nn.Parameter(checkpoint["adapter_weight"].t())
    adapter = adapter.to(device).to(clip_model.dtype)

    prompt_adapter = nn.Linear(checkpoint["prompt_weight"].shape[1], checkpoint["prompt_weight"].shape[0], bias=False)
    prompt_adapter.weight = nn.Parameter(checkpoint["prompt_weight"].t())
    prompt_adapter = prompt_adapter.to(device).to(clip_model.dtype)

    return clip_model, adapter, prompt_adapter, preprocess, checkpoint["config"]


def predict(image_path, clip_model, adapter, prompt_adapter, preprocess, config, class_names):
    device = next(clip_model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    n_class = len(class_names)
    bbeta = config.get("bbeta", 0.5)
    gamma = config.get("gamma", 0.5)
    alpha = config.get("alpha", 0.3)

    with torch.no_grad():
        features = clip_model.encode_image(image_tensor)
        features /= features.norm(dim=-1, keepdim=True)

        affinity = adapter(features)
        v_labels = torch.eye(n_class, device=device)
        v_logits = ((-1) * (bbeta - bbeta * affinity)).exp() @ v_labels

        t_logits = 100.0 * prompt_adapter(features)
        t_logits = t_logits.reshape(t_logits.shape[0], n_class, -1)
        t_logits = gamma * t_logits.mean(dim=-1) + bbeta * t_logits.max(dim=-1)[0]

        logits = t_logits + v_logits * alpha
        probs = F.softmax(logits, dim=-1).squeeze()
        top_probs, top_idx = torch.topk(probs, min(5, n_class))

    return [
        {"class": class_names[i], "confidence": float(p * 100)}
        for p, i in zip(top_probs.cpu().numpy(), top_idx.cpu().numpy())
    ]


def visualize(image_path, results, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.imshow(Image.open(image_path))
    ax1.axis("off")
    ax1.set_title("Input Image", fontweight="bold")

    classes = [r["class"] for r in results]
    confs = [r["confidence"] for r in results]
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(classes))]

    y = np.arange(len(classes))
    ax2.barh(y, confs, color=colors)
    ax2.set_yticks(y)
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()
    ax2.set_xlabel("Confidence (%)")
    ax2.set_title("Top Predictions", fontweight="bold")
    ax2.set_xlim([0, 100])
    for i, c in enumerate(confs):
        ax2.text(c + 1, i, f"{c:.1f}%", va="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="MVPDR Inference")
    parser.add_argument("--model", required=True, help="Path to trained model (.pth)")
    parser.add_argument("--image", help="Path to single image")
    parser.add_argument("--batch", help="Path to folder for batch inference")
    parser.add_argument("--classes", required=True, help="Path to class names JSON")
    parser.add_argument("--output", default="inference_results", help="Output directory")
    args = parser.parse_args()

    with open(args.classes) as f:
        class_names = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    clip_model, adapter, prompt_adapter, preprocess, config = load_model(args.model, device)
    os.makedirs(args.output, exist_ok=True)

    if args.image:
        results = predict(args.image, clip_model, adapter, prompt_adapter, preprocess, config, class_names)
        print("\nPredictions:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['class']}: {r['confidence']:.1f}%")
        visualize(args.image, results, os.path.join(args.output, "prediction.png"))

    elif args.batch:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        images = [f for f in os.listdir(args.batch) if os.path.splitext(f.lower())[1] in exts]
        print(f"Processing {len(images)} images...")

        all_results = []
        for img_file in images:
            img_path = os.path.join(args.batch, img_file)
            results = predict(img_path, clip_model, adapter, prompt_adapter, preprocess, config, class_names)
            print(f"  {img_file}: {results[0]['class']} ({results[0]['confidence']:.1f}%)")
            visualize(img_path, results, os.path.join(args.output, f"{os.path.splitext(img_file)[0]}_pred.png"))
            all_results.append({"image": img_file, "predictions": results})

        with open(os.path.join(args.output, "predictions.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDone. Results saved to {args.output}")
    else:
        parser.error("Specify --image or --batch")


if __name__ == "__main__":
    main()
