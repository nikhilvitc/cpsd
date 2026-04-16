"""Separate web UI for RGB-T fusion pedestrian classification."""

# MODULE USAGE:
# - gradio: Create a second web interface running on a separate port.
# - app module: Reuse the same inference pipeline and predict() function.
#
import os

import gradio as gr

from app import (
    CAM_PEAK_THRESHOLD,
    DETECTION_MODE,
    MIN_BOX_AREA_RATIO,
    PED_MARGIN,
    PEDESTRIAN_THRESHOLD,
    REQUIRE_THERMAL,
    predict,
)


with gr.Blocks(title="Pedestrian Detection Web App (Separate)") as demo:
    gr.Markdown(
        """
        # Pedestrian Detection (Separate Website)
        This is a separate website instance using the same RGB-T fusion architecture.
        """
    )

    with gr.Row():
        # Same input contract as main app so users can compare behavior directly.
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
        # Reuse the shared predict() implementation to avoid logic drift between apps.
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
        - Uses the same RGB-T fusion model architecture as the original app.
        - This runs on a separate URL/port and does not replace the current app.
        """
    )


if __name__ == "__main__":
    # Keep a separate default port so this app can run alongside the main app.
    port = int(os.environ.get("SEPARATE_APP_PORT", "7874"))
    demo.launch(share=False, server_port=port, theme=gr.themes.Soft())
