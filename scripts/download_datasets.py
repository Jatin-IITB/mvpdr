"""Download and prepare benchmark datasets for MVPDR training.

Supports:
  - PlantDoc (27 classes, real-world images)
  - PlantVillage (38 classes, lab-controlled images)

Usage:
    python scripts/download_datasets.py --output data/
    python scripts/download_datasets.py --dataset plantdoc --output data/
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def download_plantdoc(output_dir):
    """Download PlantDoc dataset from GitHub."""
    dataset_dir = os.path.join(output_dir, "plantdoc")
    image_dir = os.path.join(dataset_dir, "images")

    if os.path.exists(image_dir) and len(os.listdir(image_dir)) > 0:
        n_classes = len(os.listdir(image_dir))
        print(f"PlantDoc already exists at {image_dir} ({n_classes} classes), skipping.")
        return

    print("Downloading PlantDoc dataset...")
    repo_url = "https://github.com/pratikkayal/PlantDoc-Dataset.git"

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, os.path.join(tmp, "repo")],
            check=True,
        )

        src_train = os.path.join(tmp, "repo", "train")
        src_test = os.path.join(tmp, "repo", "test")

        os.makedirs(image_dir, exist_ok=True)

        for src in [src_train, src_test]:
            if not os.path.exists(src):
                continue
            for cls_name in os.listdir(src):
                cls_src = os.path.join(src, cls_name)
                if not os.path.isdir(cls_src):
                    continue
                cls_dst = os.path.join(image_dir, cls_name)
                os.makedirs(cls_dst, exist_ok=True)
                for img in os.listdir(cls_src):
                    if img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                        src_path = os.path.join(cls_src, img)
                        dst_path = os.path.join(cls_dst, img)
                        if not os.path.exists(dst_path):
                            shutil.copy2(src_path, dst_path)

    n_classes = len(os.listdir(image_dir))
    total_images = sum(
        len(os.listdir(os.path.join(image_dir, c)))
        for c in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, c))
    )
    print(f"PlantDoc ready: {n_classes} classes, {total_images} images at {image_dir}")


def download_plantvillage(output_dir):
    """Download PlantVillage dataset.

    PlantVillage is hosted on Kaggle. This function checks for the kaggle CLI
    and downloads automatically, or prints manual instructions.
    """
    dataset_dir = os.path.join(output_dir, "plantvillage")
    image_dir = os.path.join(dataset_dir, "images")

    if os.path.exists(image_dir) and len(os.listdir(image_dir)) > 0:
        n_classes = len(os.listdir(image_dir))
        print(f"PlantVillage already exists at {image_dir} ({n_classes} classes), skipping.")
        return

    kaggle_available = shutil.which("kaggle") is not None

    if kaggle_available:
        print("Downloading PlantVillage from Kaggle...")
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    "kaggle", "datasets", "download",
                    "-d", "abdallahalidev/plantvillage-dataset",
                    "-p", tmp,
                ],
                check=True,
            )
            zip_path = os.path.join(tmp, "plantvillage-dataset.zip")
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmp)

            color_dir = None
            for root, dirs, files in os.walk(tmp):
                if "color" in dirs:
                    color_dir = os.path.join(root, "color")
                    break
                if any(d.startswith("Apple") or d.startswith("Tomato") for d in dirs):
                    color_dir = root
                    break

            if color_dir is None:
                print("ERROR: Could not find image directory in downloaded archive.")
                print(f"Check contents of {tmp}")
                return

            os.makedirs(image_dir, exist_ok=True)
            for cls_name in os.listdir(color_dir):
                cls_src = os.path.join(color_dir, cls_name)
                if os.path.isdir(cls_src):
                    shutil.copytree(cls_src, os.path.join(image_dir, cls_name))

        n_classes = len(os.listdir(image_dir))
        total_images = sum(
            len(os.listdir(os.path.join(image_dir, c)))
            for c in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, c))
        )
        print(f"PlantVillage ready: {n_classes} classes, {total_images} images at {image_dir}")
    else:
        os.makedirs(dataset_dir, exist_ok=True)
        print("Kaggle CLI not found. To download PlantVillage:")
        print("  1. pip install kaggle")
        print("  2. Place kaggle.json in ~/.kaggle/")
        print("  3. Re-run this script")
        print()
        print("Or download manually:")
        print("  1. Go to https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("  2. Download and extract the 'color' folder")
        print(f"  3. Place class folders in {image_dir}/")


def verify_dataset(output_dir, name):
    """Print dataset statistics."""
    image_dir = os.path.join(output_dir, name, "images")
    if not os.path.exists(image_dir):
        print(f"  {name}: NOT FOUND")
        return False

    classes = sorted(
        d for d in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, d))
    )
    counts = []
    for cls in classes:
        n = len([
            f for f in os.listdir(os.path.join(image_dir, cls))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ])
        counts.append(n)

    if not counts:
        print(f"  {name}: 0 classes, 0 images (directory exists but empty)")
        return False

    total = sum(counts)
    print(f"  {name}: {len(classes)} classes, {total} images "
          f"(min={min(counts)}, max={max(counts)}, avg={total // len(classes)})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download MVPDR benchmark datasets")
    parser.add_argument("--output", default="data", help="Output directory (default: data/)")
    parser.add_argument(
        "--dataset", choices=["plantdoc", "plantvillage", "all"],
        default="all", help="Which dataset to download",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.dataset in ("all", "plantdoc"):
        download_plantdoc(args.output)

    if args.dataset in ("all", "plantvillage"):
        download_plantvillage(args.output)

    print("\nDataset summary:")
    for name in ["plantdoc", "plantvillage"]:
        verify_dataset(args.output, name)

    print(f"\nSet root_path in configs to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
