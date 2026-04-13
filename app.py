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
PEDESTRIAN_THRESHOLD = float(os.environ.get("PEDESTRIAN_THRESHOLD", "0.65"))
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


def gradcam_localization(rgb_t: torch.Tensor, thm_t: torch.Tensor) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
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
        return None, None

    a = acts["v"][0]
    g = grads["v"][0]
    w = g.mean(dim=(1, 2), keepdim=True)
    cam = (w * a).sum(dim=0)
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam = cam.detach().cpu().numpy()

    cam_u8 = (cam * 255).astype(np.uint8)
    # Box from high activation area (single strongest region).
    mask = cam_u8 >= max(140, int(np.percentile(cam_u8, 95)))
    if not mask.any():
        return cam, None

    ys, xs = np.where(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return cam, box


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


def predict(rgb_image: Image.Image, thermal_image: Image.Image | None):
    if rgb_image is None:
        return "Upload an RGB image to continue.", {}, "", None, None

    # Fallback: if no thermal input is provided, derive grayscale thermal from RGB.
    if thermal_image is None:
        thermal_image = rgb_image.convert("L")
        thermal_note = "Thermal image not provided. Using RGB-derived grayscale fallback."
    else:
        thermal_note = "Using uploaded RGB + thermal pair."

    rgb_t = RGB_TF(rgb_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    thm_t = THERMAL_TF(thermal_image.convert("L")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, probs = MODEL(rgb_t, thm_t)

    ped_conf = float(probs[0, PEDESTRIAN_CLASS_IDX].item())
    is_ped = ped_conf >= PEDESTRIAN_THRESHOLD
    cam, box = gradcam_localization(rgb_t, thm_t)

    rgb_out = apply_overlay(rgb_image, cam, box, draw_box=is_ped)
    thm_out = apply_overlay(thermal_image.convert("RGB"), cam, box, draw_box=is_ped)

    status = f"PEDESTRIAN DETECTED ({ped_conf:.2%})" if is_ped else f"NO PEDESTRIAN ({ped_conf:.2%})"
    thermal_note += (
        " Fusion model mode (single pedestrian region). "
        f"Threshold={PEDESTRIAN_THRESHOLD:.2f}."
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

    predict_btn = gr.Button("Detect Pedestrian", variant="primary", size="lg")

    status_out = gr.Textbox(label="Detection Result", interactive=False)
    scores_out = gr.Label(label="Class Probabilities")
    note_out = gr.Textbox(label="Input Mode", interactive=False)
    rgb_out = gr.Image(type="pil", label="Highlighted RGB Output", height=360)
    thm_out = gr.Image(type="pil", label="Highlighted Thermal Output", height=360)

    predict_btn.click(
        fn=predict,
        inputs=[rgb_input, thermal_input],
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
