"""Web UI for RGB-T fusion pedestrian classification (no scale-aware fusion block)."""

import os

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from prepare_kaist_dataset import CLASS_NAMES, NUM_CLASSES
from rgbt_fusion_model import RGBThermalFusionNet

PEDESTRIAN_CLASS_IDX = CLASS_NAMES.index("pedestrian")
# Conservative defaults reduce false positives on non-pedestrian images.
PEDESTRIAN_THRESHOLD = float(os.environ.get("PEDESTRIAN_THRESHOLD", "0.65"))
PED_MARGIN = float(os.environ.get("PEDESTRIAN_MARGIN", "0.08"))
CAM_PEAK_THRESHOLD = float(os.environ.get("CAM_PEAK_THRESHOLD", "0.30"))
MIN_BOX_AREA_RATIO = float(os.environ.get("MIN_BOX_AREA_RATIO", "0.015"))
REQUIRE_THERMAL = os.environ.get("REQUIRE_THERMAL", "1") == "1"
DETECTION_MODE = os.environ.get("DETECTION_MODE", "strict").lower()  # recall | balanced | strict
CHECKPOINT = os.environ.get("PEDESTRIAN_CKPT", "best_model_kaist.pth")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()

RGB_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
THERMAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def load_model() -> RGBThermalFusionNet:
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}. Train with train_kaist.py or set PEDESTRIAN_CKPT."
        )
    model = RGBThermalFusionNet(num_classes=NUM_CLASSES)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Fusion model loaded on {DEVICE} | checkpoint: {CHECKPOINT}")
    return model


MODEL = load_model()


def gradcam_localization(
    rgb_t: torch.Tensor, thm_t: torch.Tensor
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, float, float]:
    acts = {}
    grads = {}

    def fwd_hook(_, __, out):
        acts["v"] = out

    def bwd_hook(_, __, grad_out):
        grads["v"] = grad_out[0]

    h1 = MODEL.attn3.register_forward_hook(fwd_hook)
    h2 = MODEL.attn3.register_full_backward_hook(bwd_hook)

    MODEL.zero_grad(set_to_none=True)
    logits, _ = MODEL(rgb_t, thm_t)
    logits[:, PEDESTRIAN_CLASS_IDX].sum().backward()

    h1.remove()
    h2.remove()

    if "v" not in acts or "v" not in grads:
        return None, None, 0.0, 0.0

    a = acts["v"][0]
    g = grads["v"][0]
    w = g.mean(dim=(1, 2), keepdim=True)
    cam = (w * a).sum(dim=0)
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam = cam.detach().cpu().numpy()
    cam_peak = float(cam.max())

    cam_u8 = (cam * 255).astype(np.uint8)
    # Box from high activation area (single strongest region).
    mask = cam_u8 >= max(140, int(np.percentile(cam_u8, 95)))
    if not mask.any():
        return cam, None, cam_peak, 0.0

    ys, xs = np.where(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    h = max(int(box[3] - box[1] + 1), 1)
    w = max(int(box[2] - box[0] + 1), 1)
    area_ratio = float((h * w) / (28.0 * 28.0))
    return cam, box, cam_peak, area_ratio


def apply_overlay(base: Image.Image, cam: np.ndarray | None, box: tuple[int, int, int, int] | None, draw_box: bool) -> Image.Image:
    out = base.convert("RGB").copy()
    if cam is not None:
        cam_img = Image.fromarray((cam * 255).astype(np.uint8), mode="L").resize(out.size)
        arr = np.asarray(cam_img, dtype=np.float32) / 255.0
        alpha = np.clip((arr - 0.55) / 0.45, 0.0, 1.0)
        heat = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
        heat[..., 0] = 255
        heat[..., 3] = (alpha * 120).astype(np.uint8)
        out = Image.alpha_composite(out.convert("RGBA"), Image.fromarray(heat, mode="RGBA")).convert("RGB")

    if draw_box and box is not None:
        draw = ImageDraw.Draw(out)
        x1, y1, x2, y2 = box
        # map from feature-map coords (28x28) to image coords by scaling
        sx = out.width / 28.0
        sy = out.height / 28.0
        bx = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
        draw.rectangle([(bx[0], bx[1]), (bx[2], bx[3])], outline=(255, 0, 0), width=5)
        draw.text((bx[0] + 4, max(0, bx[1] - 16)), "pedestrian", fill=(255, 0, 0))
    return out


def predict(
    rgb_image: Image.Image,
    thermal_image: Image.Image | None,
    pedestrian_threshold: float,
    ped_margin: float,
    cam_peak_threshold: float,
    min_box_area_ratio: float,
    detection_mode: str,
    require_thermal: bool,
):
    if rgb_image is None:
        return "Upload an RGB image to continue.", {}, "", None, None

    if thermal_image is None:
        if require_thermal:
            return (
                "NO PEDESTRIAN (thermal image required)",
                {"no_pedestrian": 1.0, "pedestrian": 0.0},
                "Upload real paired RGB + thermal images. RGB-only screenshots are rejected to avoid false positives.",
                rgb_image.convert("RGB"),
                None,
            )
        thermal_image = rgb_image.convert("L")
        thermal_note = "Thermal image not provided. Using RGB-derived grayscale fallback (less reliable)."
    else:
        thermal_note = "Using uploaded RGB + thermal pair."

    rgb_t = RGB_TF(rgb_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    thm_t = THERMAL_TF(thermal_image.convert("L")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, probs = MODEL(rgb_t, thm_t)

    ped_conf = float(probs[0, PEDESTRIAN_CLASS_IDX].item())
    no_ped_conf = float(probs[0, 1 - PEDESTRIAN_CLASS_IDX].item())
    cam, box, cam_peak, box_area_ratio = gradcam_localization(rgb_t, thm_t)

    if detection_mode == "strict":
        is_ped = (
            ped_conf >= pedestrian_threshold
            and (ped_conf - no_ped_conf) >= ped_margin
            and cam_peak >= cam_peak_threshold
            and box_area_ratio >= min_box_area_ratio
        )
    elif detection_mode == "balanced":
        is_ped = (
            ped_conf >= pedestrian_threshold
            and (ped_conf - no_ped_conf) >= ped_margin
            and (cam_peak >= cam_peak_threshold or box_area_ratio >= min_box_area_ratio)
        )
    else:
        # recall mode: prioritize detecting pedestrians
        is_ped = ped_conf >= pedestrian_threshold

    rgb_out = apply_overlay(rgb_image, cam, box, draw_box=is_ped)
    thm_out = apply_overlay(thermal_image.convert("RGB"), cam, box, draw_box=is_ped)

    status = f"PEDESTRIAN DETECTED ({ped_conf:.2%})" if is_ped else f"NO PEDESTRIAN ({ped_conf:.2%})"
    thermal_note += (
        f" Fusion model mode (single pedestrian region, mode={detection_mode}). "
        f"Threshold={pedestrian_threshold:.2f}, margin={ped_margin:.2f}, "
        f"cam_peak={cam_peak:.2f}, box_ratio={box_area_ratio:.3f}."
    )
    scores = {
        "no_pedestrian": round(float(probs[0, 0].item()), 4),
        "pedestrian": round(float(probs[0, 1].item()), 4),
    }
    return status, scores, thermal_note, rgb_out, thm_out


with gr.Blocks(title="Pedestrian Detection Web App") as demo:
    gr.Markdown(
        """
        # Pedestrian Detection (RGB-T Fusion)
        Upload an RGB image and optionally a thermal image.
        Uses the original RGB-T cross-modal attention fusion model.
        """
    )

    with gr.Row():
        rgb_input = gr.Image(type="pil", label="RGB Image", height=320)
        thermal_input = gr.Image(type="pil", label="Thermal Image (Optional)", height=320)

    with gr.Row():
        detection_mode_input = gr.Dropdown(
            choices=["strict", "balanced", "recall"],
            value=DETECTION_MODE if DETECTION_MODE in {"strict", "balanced", "recall"} else "strict",
            label="Detection Mode",
        )
        require_thermal_input = gr.Checkbox(value=REQUIRE_THERMAL, label="Require thermal image")

    with gr.Row():
        pedestrian_threshold_input = gr.Slider(0.0, 1.0, value=PEDESTRIAN_THRESHOLD, step=0.01, label="Pedestrian Threshold")
        ped_margin_input = gr.Slider(0.0, 0.5, value=PED_MARGIN, step=0.01, label="Margin over non-pedestrian")

    with gr.Row():
        cam_peak_threshold_input = gr.Slider(0.0, 1.0, value=CAM_PEAK_THRESHOLD, step=0.01, label="CAM Peak Threshold")
        min_box_area_ratio_input = gr.Slider(0.0, 0.2, value=MIN_BOX_AREA_RATIO, step=0.001, label="Minimum Box Area Ratio")

    predict_btn = gr.Button("Detect Pedestrian", variant="primary", size="lg")

    status_out = gr.Textbox(label="Detection Result", interactive=False)
    scores_out = gr.Label(label="Class Probabilities")
    note_out = gr.Textbox(label="Input Mode", interactive=False)
    rgb_out = gr.Image(type="pil", label="Highlighted RGB Output", height=360)
    thm_out = gr.Image(type="pil", label="Highlighted Thermal Output", height=360)

    predict_btn.click(
        fn=predict,
        inputs=[
            rgb_input,
            thermal_input,
            pedestrian_threshold_input,
            ped_margin_input,
            cam_peak_threshold_input,
            min_box_area_ratio_input,
            detection_mode_input,
            require_thermal_input,
        ],
        outputs=[status_out, scores_out, note_out, rgb_out, thm_out],
    )

    gr.Markdown(
        """
        ### Notes
        - This is the fusion classifier architecture (no scale-aware fusion block).
        - Localization is Grad-CAM based from the fusion model.
        - It gives one main pedestrian region, not multi-person detector boxes.
        """
    )


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
