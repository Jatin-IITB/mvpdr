import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import json

def load_trained_model(model_path, device='cuda'):
    """Load a trained MVPDR model"""
    checkpoint = torch.load(model_path, map_location=device)

    # Load CLIP backbone
    clip_model, preprocess = clip.load(checkpoint['clip_backbone'], device=device)
    clip_model.eval()

    # Load adapters
    adapter = nn.Linear(checkpoint['adapter_weight'].shape[1], 
                       checkpoint['adapter_weight'].shape[0], bias=False)
    adapter.weight = nn.Parameter(checkpoint['adapter_weight'].t())
    adapter = adapter.to(device).to(clip_model.dtype)

    prompt_adapter = nn.Linear(checkpoint['prompt_weight'].shape[1],
                               checkpoint['prompt_weight'].shape[0], bias=False)
    prompt_adapter.weight = nn.Parameter(checkpoint['prompt_weight'].t())
    prompt_adapter = prompt_adapter.to(device).to(clip_model.dtype)

    return clip_model, adapter, prompt_adapter, preprocess, checkpoint['config']


def predict_single_image(image_path, clip_model, adapter, prompt_adapter, 
                         preprocess, config, class_names):
    """Predict disease for a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).cuda()

    # Extract features
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Visual logits
        affinity = adapter(image_features)

        # For visual logits, we need the labels matrix (one-hot encoded)
        n_class = len(class_names)
        # Create identity matrix as placeholder for labels
        v_labels = torch.eye(n_class).cuda()

        bbeta = config.get('bbeta', 0.5)
        v_logits = ((-1) * (bbeta - bbeta * affinity)).exp() @ v_labels

        # Textual logits
        t_logits = 100. * prompt_adapter(image_features)
        t_logits = t_logits.reshape(t_logits.shape[0], n_class, -1)
        t_mean_logits = t_logits.mean(dim=-1)
        t_max_logits = t_logits.max(dim=-1)[0]

        gamma = config.get('gamma', 0.5)
        t_logits_combined = gamma * t_mean_logits + bbeta * t_max_logits

        # Final prediction
        alpha = config.get('alpha', 0.3)
        final_logits = t_logits_combined + v_logits * alpha

        # Get probabilities
        probs = F.softmax(final_logits, dim=-1).squeeze()

        # Get top-5 predictions
        top5_probs, top5_indices = torch.topk(probs, min(5, len(class_names)))

    results = []
    for prob, idx in zip(top5_probs.cpu().numpy(), top5_indices.cpu().numpy()):
        results.append({
            'class': class_names[idx],
            'confidence': float(prob * 100)
        })

    return results, probs.cpu().numpy()


def visualize_prediction(image_path, results, save_path=None):
    """Visualize prediction results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Display image
    img = Image.open(image_path)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')

    # Display top predictions
    classes = [r['class'] for r in results]
    confidences = [r['confidence'] for r in results]

    colors = ['green' if i == 0 else 'lightblue' for i in range(len(classes))]
    y_pos = np.arange(len(classes))

    ax2.barh(y_pos, confidences, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Top-5 Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 100])

    # Add confidence values on bars
    for i, (cls, conf) in enumerate(zip(classes, confidences)):
        ax2.text(conf + 1, i, f'{conf:.2f}%', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def batch_inference(image_folder, model_path, class_names, output_dir):
    """Run inference on a folder of images"""
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    clip_model, adapter, prompt_adapter, preprocess, config = load_trained_model(model_path)

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f.lower())[1] in image_extensions]

    print(f"Found {len(image_files)} images")

    all_results = []

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        print(f"\nProcessing: {img_file}")

        try:
            results, _ = predict_single_image(
                img_path, clip_model, adapter, prompt_adapter,
                preprocess, config, class_names
            )

            print(f"  Prediction: {results[0]['class']} ({results[0]['confidence']:.2f}%)")

            # Save visualization
            vis_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_prediction.png")
            visualize_prediction(img_path, results, vis_path)

            all_results.append({
                'image': img_file,
                'predictions': results
            })

        except Exception as e:
            print(f"  Error processing {img_file}: {e}")

    # Save all results to JSON
    results_file = os.path.join(output_dir, 'batch_predictions.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\nBatch inference complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='MVPDR Inference Script')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--image', type=str, help='Path to single image for inference')
    parser.add_argument('--batch', type=str, help='Path to folder for batch inference')
    parser.add_argument('--output', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--classes', type=str, required=True, help='Path to class names JSON file')

    args = parser.parse_args()

    # Load class names
    with open(args.classes, 'r') as f:
        class_names = json.load(f)

    if args.image:
        # Single image inference
        print("Loading model...")
        clip_model, adapter, prompt_adapter, preprocess, config = load_trained_model(args.model)

        print(f"Running inference on: {args.image}")
        results, _ = predict_single_image(
            args.image, clip_model, adapter, prompt_adapter,
            preprocess, config, class_names
        )

        print("\nTop-5 Predictions:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['class']}: {result['confidence']:.2f}%")

        # Visualize
        os.makedirs(args.output, exist_ok=True)
        vis_path = os.path.join(args.output, 'prediction.png')
        visualize_prediction(args.image, results, vis_path)

    elif args.batch:
        # Batch inference
        batch_inference(args.batch, args.model, class_names, args.output)

    else:
        print("Error: Please specify either --image or --batch")


if __name__ == '__main__':
    main()
