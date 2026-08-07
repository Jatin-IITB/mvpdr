"""MVPDR+ Interactive Demo — Gradio web interface.

Upload a plant leaf image to get:
  - Disease classification with confidence scores
  - GradCAM heatmap showing which regions drive the prediction
  - Open-set confidence (is this a known disease?)
  - Prototype attention weights (when cross-attention is enabled)

Usage:
    python app.py --config configs/plantdoc_plus.yaml \\
                  --checkpoint results/.../mvpdr_plus_model.pth

    # Without a checkpoint (zero-shot CLIP baseline):
    python app.py --config configs/plantdoc_plus.yaml --zero_shot
"""

import argparse
import os
import tempfile
import uuid

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from mvpdr import clip
from mvpdr.agent import DiagnosticContext, run_agentic_diagnosis
from mvpdr.interpretability import GradCAM, plot_gradcam, plot_prototype_attention
from mvpdr.knowledge import DiseaseKnowledgeBase
from mvpdr.models import MVPDRPlus
from mvpdr.openset import energy_score, msp_score
from mvpdr.severity import estimate_severity_heuristic, plot_severity
from mvpdr.utils import clip_classifier


# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------
_state = {}


def load_model(config_path, checkpoint_path=None, zero_shot=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    clip_model, preprocess = clip.load(cfg["backbone"])
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    if checkpoint_path and not zero_shot:
        ckpt = torch.load(checkpoint_path, weights_only=True, map_location=device)
        classnames = ckpt.get("classnames")
        if classnames is None:
            from mvpdr.datasets import build_dataset
            dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])
            classnames = dataset.classnames

        cfg["n_classes"] = len(classnames)
        model = MVPDRPlus(clip_model, classnames, cfg).to(device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.eval()
        mode = "mvpdr_plus"
    else:
        from mvpdr.datasets import build_dataset
        dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])
        classnames = dataset.classnames
        cfg["n_classes"] = len(classnames)
        model = None
        mode = "zero_shot"

    _state["clip_model"] = clip_model
    _state["preprocess"] = preprocess
    _state["model"] = model
    _state["classnames"] = classnames
    _state["cfg"] = cfg
    _state["device"] = device
    _state["mode"] = mode

    _state["knowledge_base"] = DiseaseKnowledgeBase()

    print(f"Loaded: {cfg['dataset']} | {cfg['backbone']} | "
          f"{'MVPDRPlus' if model else 'Zero-shot CLIP'} | {device}")
    print(f"Classes: {len(classnames)}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(image_pil):
    if image_pil is None:
        return None, "No image provided", None, "", None, ""

    clip_model = _state["clip_model"]
    preprocess = _state["preprocess"]
    model = _state["model"]
    classnames = _state["classnames"]
    device = _state["device"]
    cfg = _state["cfg"]

    image_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features = F.normalize(image_features, dim=-1)

    # ---- classification ----
    if model is not None:
        with torch.no_grad():
            logits, aux = model(image_features, clip_model)
    else:
        template = ["a photo of a {}."]
        text_weights = clip_classifier(classnames, template, clip_model)
        logits = 100.0 * image_features.float() @ text_weights.float()
        aux = {}

    probs = F.softmax(logits, dim=-1).squeeze(0)
    top_k = min(7, len(classnames))
    top_probs, top_idx = torch.topk(probs, top_k)

    # ---- confidence bar chart ----
    fig_conf, ax = plt.subplots(figsize=(8, 4))
    names = [classnames[i] for i in top_idx.cpu().numpy()]
    confs = (top_probs.cpu().numpy() * 100).tolist()
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(names))]

    y = np.arange(len(names))
    ax.barh(y, confs, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.set_xlim(0, 100)
    for i, c in enumerate(confs):
        ax.text(c + 1, i, f"{c:.1f}%", va="center", fontsize=10)
    ax.set_title("Top Predictions", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()

    conf_path = os.path.join(tempfile.gettempdir(), f"mvpdr_conf_{uuid.uuid4().hex[:8]}.png")
    fig_conf.savefig(conf_path, dpi=150, bbox_inches="tight")
    plt.close(fig_conf)

    pred_name = classnames[top_idx[0].item()]
    pred_conf = top_probs[0].item() * 100

    # ---- GradCAM ----
    if model is not None and model.prompt_learner is not None:
        with torch.no_grad():
            text_features = model.prompt_learner(clip_model)
            text_features = F.normalize(text_features, dim=-1)
    else:
        template = ["a photo of a {}."]
        text_features = clip_classifier(classnames, template, clip_model).t()

    cam_resized = None
    gradcam = GradCAM(clip_model)
    try:
        cam, cam_idx = gradcam.generate(image_tensor, text_features, class_idx=top_idx[0].item())
        cam_resized = np.array(
            Image.fromarray(np.uint8(cam * 255)).resize(
                image_pil.size, resample=Image.BILINEAR,
            )
        ) / 255.0

        fig_cam, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image_pil)
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")
        axes[1].imshow(cam_resized, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("GradCAM", fontsize=12)
        axes[1].axis("off")
        axes[2].imshow(image_pil)
        axes[2].imshow(cam_resized, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title("Overlay", fontsize=12)
        axes[2].axis("off")
        fig_cam.suptitle(f"Activation: {pred_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        cam_path = os.path.join(tempfile.gettempdir(), f"mvpdr_cam_{uuid.uuid4().hex[:8]}.png")
        fig_cam.savefig(cam_path, dpi=150, bbox_inches="tight")
        plt.close(fig_cam)
    except RuntimeError:
        cam_path = None
    finally:
        gradcam.remove_hooks()

    # ---- severity estimation ----
    severity = estimate_severity_heuristic(
        gradcam_map=cam_resized if cam_resized is not None else np.zeros((1, 1)),
        classification_confidence=top_probs[0].item(),
    )

    sev_path = os.path.join(tempfile.gettempdir(), f"mvpdr_sev_{uuid.uuid4().hex[:8]}.png")
    try:
        plot_severity(severity, save_path=sev_path)
    except Exception:
        sev_path = None

    # ---- open-set score ----
    e_score = energy_score(logits).item()
    m_score = msp_score(logits).item()
    openset_text = (
        f"**Energy Score:** {e_score:.2f}\n"
        f"**Max Softmax Prob:** {m_score:.3f}\n\n"
    )
    if m_score > 0.5:
        openset_text += "The model is **confident** this is a known disease class."
    elif m_score > 0.2:
        openset_text += "The model has **moderate** confidence — verify manually."
    else:
        openset_text += "**Low confidence** — this may be an unknown class or non-disease image."

    # ---- diagnostic report ----
    top_predictions = [
        (classnames[idx.item()], prob.item())
        for idx, prob in zip(top_idx, top_probs)
    ]

    ctx = DiagnosticContext(
        classnames=classnames,
        top_predictions=top_predictions,
        severity=severity,
        knowledge_base=_state.get("knowledge_base"),
    )

    report = run_agentic_diagnosis(ctx)
    report_md = report.to_markdown()

    # ---- result summary ----
    summary = f"**{pred_name}** ({pred_conf:.1f}% confidence)"

    return conf_path, summary, cam_path, openset_text, sev_path, report_md


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo():
    with gr.Blocks(
        title="MVPDR+ Plant Disease Diagnostic System",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# MVPDR+ Plant Disease Diagnostic System\n"
            "Upload a plant leaf image for AI-powered disease diagnosis with "
            "severity estimation, treatment recommendations, and visual explanations.\n\n"
            "Powered by CLIP-based multi-view prototype recognition with learnable prompts, "
            "hierarchical prototypes, cross-attention fusion, and an agentic diagnostic pipeline."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Plant Leaf Image",
                    type="pil",
                    height=350,
                )
                run_btn = gr.Button("Diagnose", variant="primary", size="lg")

                mode_info = _state.get("mode", "unknown")
                model_name = "MVPDRPlus" if mode_info == "mvpdr_plus" else "Zero-shot CLIP"
                backbone = _state.get("cfg", {}).get("backbone", "?")
                dataset = _state.get("cfg", {}).get("dataset", "?")
                n_cls = len(_state.get("classnames", []))
                try:
                    import ollama as _ol
                    agent_mode = "Agentic (Ollama/Qwen)"
                except ImportError:
                    agent_mode = "Agentic (Claude)" if os.environ.get("ANTHROPIC_API_KEY") else "Rule-based"
                gr.Markdown(
                    f"**Model:** {model_name}  \n"
                    f"**Backbone:** {backbone}  \n"
                    f"**Dataset:** {dataset} ({n_cls} classes)  \n"
                    f"**Agent:** {agent_mode}"
                )

            with gr.Column(scale=2):
                summary_out = gr.Markdown(label="Prediction")
                conf_out = gr.Image(label="Confidence Scores", height=280)

        with gr.Tabs():
            with gr.TabItem("Visual Explanation"):
                with gr.Row():
                    cam_out = gr.Image(label="GradCAM Visualization", height=320)
                    sev_out = gr.Image(label="Severity Assessment", height=320)
            with gr.TabItem("Diagnostic Report"):
                report_out = gr.Markdown(label="AI Diagnostic Report")
            with gr.TabItem("Open-Set Detection"):
                openset_out = gr.Markdown(label="Open-Set Detection Scores")

        run_btn.click(
            fn=predict,
            inputs=[image_input],
            outputs=[conf_out, summary_out, cam_out, openset_out, sev_out, report_out],
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MVPDR+ Gradio Demo")
    parser.add_argument("--config", default="configs/plantdoc_plus.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    load_model(args.config, args.checkpoint, args.zero_shot)
    demo = build_demo()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
