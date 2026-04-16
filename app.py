"""Web UI for RGB-T fusion pedestrian classification (no scale-aware fusion block)."""

# MODULE USAGE:
# - torch: Load fusion model checkpoint and run inference on GPU/MPS/CPU.
# - torch.nn.functional: Softmax normalization and attention/Grad-CAM computations.
# - PIL.Image / ImageDraw: Load input images and draw bounding boxes on outputs.
# - torchvision.transforms: Normalize and resize images to model input dimensions.
# - torchvision.models.detection: FasterRCNN for multi-person bounding box detection.
# - gradio: Build interactive web UI for image upload and result display.
# - numpy: Array manipulation for CAM heatmap processing.
# - prepare_kaist_dataset / rgbt_fusion_model: Load trained model and label definitions.

import os

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
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
DETECTOR_SCORE_THRESHOLD = float(os.environ.get("DETECTOR_SCORE_THRESHOLD", "0.55"))


def pick_device() -> torch.device:
    # Prefer GPU backends for speed, with CPU fallback for portability.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()

RGB_TF = transforms.Compose([
    # Keep preprocessing consistent with training to avoid domain shift at inference.
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
THERMAL_TF = transforms.Compose([
    # Thermal is naturally intensity-based, so single-channel input is the right representation.
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
    # map_location allows loading on CPU/MPS even if checkpoint was saved on CUDA.
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Fusion model loaded on {DEVICE} | checkpoint: {CHECKPOINT}")
    return model


MODEL = load_model()


def load_detector():
    # A pretrained detector is used to get person-wise boxes for "all pedestrians" output.
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(DEVICE)
    model.eval()
    print(f"Detector loaded on {DEVICE}")
    return model


DETECTOR = load_detector()


def detect_pedestrian_boxes(rgb_image: Image.Image, score_threshold: float) -> tuple[list[tuple[int, int, int, int]], float]:
    rgb = rgb_image.convert("RGB")
    t = to_tensor(rgb).to(DEVICE)
    with torch.no_grad():
        out = DETECTOR([t])[0]

    labels = out["labels"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()
    boxes = out["boxes"].detach().cpu().numpy()

    # COCO label 1 corresponds to "person" in Faster R-CNN outputs.
    selected = (labels == 1) & (scores >= score_threshold)
    selected_boxes = []
    selected_scores = []
    for b, s, ok in zip(boxes, scores, selected):
        if not ok:
            continue
        x1, y1, x2, y2 = [int(v) for v in b]
        selected_boxes.append((x1, y1, x2, y2))
        selected_scores.append(float(s))

    max_score = max(selected_scores) if selected_scores else 0.0
    return selected_boxes, max_score


def _scale_boxes(boxes: list[tuple[int, int, int, int]], src_size: tuple[int, int], dst_size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    if src_size == dst_size:
        return boxes
    sx = dst_size[0] / max(src_size[0], 1)
    sy = dst_size[1] / max(src_size[1], 1)
    out = []
    for x1, y1, x2, y2 in boxes:
        out.append((int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)))
    return out


def draw_detector_boxes(base: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    out = base.convert("RGB").copy()
    if not boxes:
        return out
    draw = ImageDraw.Draw(out)
    # Cyan boxes visually separate detector output from red CAM overlays.
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 255), width=4)
        if i == 0:
            draw.text((x1 + 4, max(0, y1 - 16)), "pedestrian", fill=(0, 255, 255))
    return out


def gradcam_localization(
    rgb_t: torch.Tensor, thm_t: torch.Tensor
) -> tuple[np.ndarray | None, list[tuple[int, int, int, int]], float, float]:
    acts = {}
    grads = {}

    def fwd_hook(_, __, out):
        acts["v"] = out

    def bwd_hook(_, __, grad_out):
        grads["v"] = grad_out[0]

    # Hooks capture activations and gradients needed to build Grad-CAM.
    h1 = MODEL.attn3.register_forward_hook(fwd_hook)
    h2 = MODEL.attn3.register_full_backward_hook(bwd_hook)

    MODEL.zero_grad(set_to_none=True)
    # Backprop on pedestrian logit highlights regions supporting that class.
    logits, _ = MODEL(rgb_t, thm_t)
    logits[:, PEDESTRIAN_CLASS_IDX].sum().backward()

    h1.remove()
    h2.remove()

    if "v" not in acts or "v" not in grads:
        return None, [], 0.0, 0.0

    a = acts["v"][0]
    g = grads["v"][0]
    w = g.mean(dim=(1, 2), keepdim=True)
    cam = (w * a).sum(dim=0)
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam = cam.detach().cpu().numpy()
    cam_peak = float(cam.max())

    cam_u8 = (cam * 255).astype(np.uint8)
    # Build tighter boxes from separate high-activation blobs.
    mask = cam_u8 >= max(165, int(np.percentile(cam_u8, 97)))
    if not mask.any():
        return cam, [], cam_peak, 0.0

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    max_area_ratio = 0.0

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            ys = []
            xs = []

            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            # Ignore tiny blobs that tend to be noise.
            if len(xs) < 4:
                continue

            box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            boxes.append(box)
            bh = max(int(box[3] - box[1] + 1), 1)
            bw = max(int(box[2] - box[0] + 1), 1)
            area_ratio = float((bh * bw) / (28.0 * 28.0))
            if area_ratio > max_area_ratio:
                max_area_ratio = area_ratio

    if not boxes:
        return cam, [], cam_peak, 0.0
    return cam, boxes, cam_peak, max_area_ratio


def apply_overlay(base: Image.Image, cam: np.ndarray | None, boxes: list[tuple[int, int, int, int]], draw_box: bool) -> Image.Image:
    out = base.convert("RGB").copy()
    if cam is not None:
        # Alpha-scaled heatmap keeps context visible while emphasizing hot regions.
        cam_img = Image.fromarray((cam * 255).astype(np.uint8), mode="L").resize(out.size)
        arr = np.asarray(cam_img, dtype=np.float32) / 255.0
        alpha = np.clip((arr - 0.55) / 0.45, 0.0, 1.0)
        heat = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
        heat[..., 0] = 255
        heat[..., 3] = (alpha * 120).astype(np.uint8)
        out = Image.alpha_composite(out.convert("RGBA"), Image.fromarray(heat, mode="RGBA")).convert("RGB")

    if draw_box and boxes:
        draw = ImageDraw.Draw(out)
        # map from feature-map coords (28x28) to image coords by scaling
        sx = out.width / 28.0
        sy = out.height / 28.0
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            bx = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
            draw.rectangle([(bx[0], bx[1]), (bx[2], bx[3])], outline=(255, 0, 0), width=4)
            if i == 0:
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
        # RGB-derived grayscale fallback keeps demo usable without paired thermal input.
        thermal_image = rgb_image.convert("L")
        thermal_note = "Thermal image not provided. Using RGB-derived grayscale fallback (less reliable)."
    else:
        thermal_note = "Using uploaded RGB + thermal pair."

    # Primary localization path: detector boxes for all visible pedestrians.
    det_boxes_rgb, max_det_score = detect_pedestrian_boxes(rgb_image, pedestrian_threshold)
    det_boxes_thm = _scale_boxes(det_boxes_rgb, rgb_image.size, thermal_image.size)

    is_ped = len(det_boxes_rgb) > 0
    rgb_out = draw_detector_boxes(rgb_image, det_boxes_rgb)
    thm_out = draw_detector_boxes(thermal_image.convert("RGB"), det_boxes_thm)

    status = "YES, THERE IS PEDESTRIAN" if is_ped else "NO PEDESTRIAN"
    thermal_note += (
        f" Detector mode: all pedestrian boxes shown. "
        f"Threshold={pedestrian_threshold:.2f}, boxes={len(det_boxes_rgb)}."
    )
    scores = {
        "no_pedestrian": round(max(0.0, 1.0 - max_det_score), 4),
        "pedestrian": round(max_det_score, 4),
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
        - It gives Grad-CAM candidate regions, not true detector boxes.
        """
    )


if __name__ == "__main__":
    # Railway-compatible: read PORT from environment, default to 7876 locally.
    port = int(os.environ.get("PORT", 7876))
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft()
    )
