"""Run the complete MVPDR experiment matrix.

Orchestrates all training runs:
  1. Zero-shot CLIP baselines (all backbones × all datasets)
  2. MVPDR baseline (1/5/10/20 shots × datasets)
  3. MVPDR+ full model (1/5/10/20 shots × datasets)
  4. MVPDR+ ablation study (component toggles on PlantDoc 20-shot)

Usage:
    # Full experiment suite
    python scripts/run_experiments.py --root_path data/ --seeds 1 2 3

    # Single stage
    python scripts/run_experiments.py --root_path data/ --stage zeroshot
    python scripts/run_experiments.py --root_path data/ --stage mvpdr --shots 20
    python scripts/run_experiments.py --root_path data/ --stage mvpdr_plus --dataset plantdoc
    python scripts/run_experiments.py --root_path data/ --stage ablation
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASETS = ["plantdoc", "plantvillage"]
BACKBONES = ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]
SHOT_SETTINGS = [1, 5, 10, 20]
DEFAULT_BACKBONE = "RN101"

ABLATION_CONFIGS = [
    {"name": "coop_only", "use_prompt_learner": True, "use_prototype_bank": False, "use_cross_attn": False},
    {"name": "proto_only", "use_prompt_learner": False, "use_prototype_bank": True, "use_cross_attn": False},
    {"name": "crossattn_only", "use_prompt_learner": False, "use_prototype_bank": False, "use_cross_attn": True},
    {"name": "coop_proto", "use_prompt_learner": True, "use_prototype_bank": True, "use_cross_attn": False},
    {"name": "coop_crossattn", "use_prompt_learner": True, "use_prototype_bank": False, "use_cross_attn": True},
    {"name": "proto_crossattn", "use_prompt_learner": False, "use_prototype_bank": True, "use_cross_attn": True},
    {"name": "full", "use_prompt_learner": True, "use_prototype_bank": True, "use_cross_attn": True},
]


def run_command(cmd, dry_run=False):
    cmd_str = " ".join(cmd)
    print(f"  >> {cmd_str}")
    if dry_run:
        return 0
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def update_config(config_path, overrides, tmp_dir):
    """Create a temporary config with overrides applied."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg.update(overrides)

    name_parts = [cfg["dataset"]]
    for k, v in overrides.items():
        if k not in ("root_path", "load_cache", "load_pre_feat"):
            name_parts.append(f"{k}={v}")
    tmp_name = "_".join(str(p) for p in name_parts) + ".yaml"
    tmp_path = os.path.join(tmp_dir, tmp_name)

    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return tmp_path


def stage_zeroshot(args):
    """Run zero-shot evaluation across all datasets and backbones."""
    print("\n" + "=" * 60)
    print("STAGE 1: Zero-Shot CLIP Baselines")
    print("=" * 60)

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "evaluate_zeroshot.py"),
        "--root_path", args.root_path,
        "--output", os.path.join(args.output, "zeroshot"),
    ]
    if args.backbone != "all":
        cmd.extend(["--backbone", args.backbone])
    if args.dataset != "all":
        cmd.extend(["--dataset", args.dataset])
    cmd.extend(["--gpu", args.gpu])

    return run_command(cmd, args.dry_run)


def stage_mvpdr(args):
    """Run MVPDR baseline training."""
    print("\n" + "=" * 60)
    print("STAGE 2: MVPDR Baseline Training")
    print("=" * 60)

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    shots = args.shots_list
    backbone = args.backbone if args.backbone != "all" else DEFAULT_BACKBONE

    for ds in datasets:
        config_path = str(REPO_ROOT / "configs" / f"{ds}.yaml")
        if not os.path.exists(config_path):
            print(f"  Skipping {ds}: no config at {config_path}")
            continue

        for n_shots in shots:
            for seed in args.seeds:
                print(f"\n--- MVPDR | {ds} | {backbone} | {n_shots}-shot | seed={seed} ---")

                tmp_cfg = update_config(config_path, {
                    "root_path": args.root_path,
                    "shots": n_shots,
                }, args.tmp_dir)

                cmd = [
                    sys.executable, str(REPO_ROOT / "train.py"),
                    "--config", tmp_cfg,
                    "--backbone", backbone,
                    "--seed", str(seed),
                    "--gpu", args.gpu,
                ]
                rc = run_command(cmd, args.dry_run)
                if rc != 0:
                    print(f"  FAILED (exit code {rc})")


def stage_mvpdr_plus(args):
    """Run MVPDR+ training."""
    print("\n" + "=" * 60)
    print("STAGE 3: MVPDR+ Training")
    print("=" * 60)

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    shots = args.shots_list
    backbone = args.backbone if args.backbone != "all" else DEFAULT_BACKBONE

    for ds in datasets:
        config_path = str(REPO_ROOT / "configs" / f"{ds}_plus.yaml")
        if not os.path.exists(config_path):
            print(f"  Skipping {ds}: no config at {config_path}")
            continue

        for n_shots in shots:
            for seed in args.seeds:
                print(f"\n--- MVPDR+ | {ds} | {backbone} | {n_shots}-shot | seed={seed} ---")

                tmp_cfg = update_config(config_path, {
                    "root_path": args.root_path,
                    "shots": n_shots,
                }, args.tmp_dir)

                cmd = [
                    sys.executable, str(REPO_ROOT / "train_plus.py"),
                    "--config", tmp_cfg,
                    "--backbone", backbone,
                    "--shots", str(n_shots),
                    "--seed", str(seed),
                    "--gpu", args.gpu,
                ]
                rc = run_command(cmd, args.dry_run)
                if rc != 0:
                    print(f"  FAILED (exit code {rc})")


def stage_ablation(args):
    """Run MVPDR+ ablation study on PlantDoc 20-shot."""
    print("\n" + "=" * 60)
    print("STAGE 4: MVPDR+ Ablation Study (PlantDoc 20-shot)")
    print("=" * 60)

    config_path = str(REPO_ROOT / "configs" / "plantdoc_plus.yaml")
    backbone = args.backbone if args.backbone != "all" else DEFAULT_BACKBONE

    for ablation in ABLATION_CONFIGS:
        name = ablation["name"]
        for seed in args.seeds:
            print(f"\n--- Ablation: {name} | seed={seed} ---")

            overrides = {
                "root_path": args.root_path,
                "shots": 20,
                "use_prompt_learner": ablation["use_prompt_learner"],
                "use_prototype_bank": ablation["use_prototype_bank"],
                "use_cross_attn": ablation["use_cross_attn"],
            }
            tmp_cfg = update_config(config_path, overrides, args.tmp_dir)

            out_dir = os.path.join(
                args.output, "ablation", name, f"seed{seed}",
            )

            cmd = [
                sys.executable, str(REPO_ROOT / "train_plus.py"),
                "--config", tmp_cfg,
                "--backbone", backbone,
                "--shots", "20",
                "--seed", str(seed),
                "--gpu", args.gpu,
                "--output_dir", out_dir,
            ]
            rc = run_command(cmd, args.dry_run)
            if rc != 0:
                print(f"  FAILED (exit code {rc})")


def stage_backbone_comparison(args):
    """Run MVPDR+ with different backbones on PlantDoc 20-shot."""
    print("\n" + "=" * 60)
    print("STAGE 5: Backbone Comparison (PlantDoc 20-shot)")
    print("=" * 60)

    config_path = str(REPO_ROOT / "configs" / "plantdoc_plus.yaml")

    for backbone in BACKBONES:
        for seed in args.seeds:
            print(f"\n--- Backbone: {backbone} | seed={seed} ---")

            tmp_cfg = update_config(config_path, {
                "root_path": args.root_path,
                "shots": 20,
            }, args.tmp_dir)

            cmd = [
                sys.executable, str(REPO_ROOT / "train_plus.py"),
                "--config", tmp_cfg,
                "--backbone", backbone,
                "--shots", "20",
                "--seed", str(seed),
                "--gpu", args.gpu,
            ]
            rc = run_command(cmd, args.dry_run)
            if rc != 0:
                print(f"  FAILED (exit code {rc})")


STAGES = {
    "zeroshot": stage_zeroshot,
    "mvpdr": stage_mvpdr,
    "mvpdr_plus": stage_mvpdr_plus,
    "ablation": stage_ablation,
    "backbone": stage_backbone_comparison,
}


def main():
    parser = argparse.ArgumentParser(description="MVPDR Full Experiment Suite")
    parser.add_argument("--root_path", required=True, help="Dataset root directory")
    parser.add_argument("--output", default="results", help="Results output directory")
    parser.add_argument(
        "--stage", default="all",
        choices=list(STAGES.keys()) + ["all"],
        help="Which stage to run",
    )
    parser.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    parser.add_argument("--backbone", default="all", choices=BACKBONES + ["all"])
    parser.add_argument("--shots", type=int, nargs="+", default=None,
                        help="Shot settings (default: 1 5 10 20)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1],
                        help="Random seeds (default: 1)")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()

    args.shots_list = args.shots or SHOT_SETTINGS
    args.tmp_dir = os.path.join(args.output, ".tmp_configs")
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    start = time.time()

    if args.stage == "all":
        for name, fn in STAGES.items():
            fn(args)
    else:
        STAGES[args.stage](args)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Total time: {elapsed / 3600:.1f} hours ({elapsed / 60:.0f} min)")
    print(f"Results in: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
