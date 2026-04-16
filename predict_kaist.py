# MODULE USAGE:
# - argparse: Parse command-line arguments (--rgb, --thermal, --checkpoint, --img-size).
# - torch: Load model checkpoint, handle device placement (CUDA/MPS/CPU), and run inference.
# - PIL.Image: Load image files from disk for preprocessing.
# - torchvision.transforms: Normalize and resize images to match training dimensions.

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from prepare_kaist_dataset import CLASS_NAMES, NUM_CLASSES
from rgbt_fusion_model import RGBThermalFusionNet


# Same normalization used in training so inference features stay consistent.
RGB_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Thermal is single-channel heat intensity; grayscale avoids pseudo-color bias.
THM_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb", required=True)
    p.add_argument("--thermal", required=True)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--checkpoint", default="best_model_kaist.pth")
    args = p.parse_args()

    # Device fallback allows the same command on GPU/MPS/CPU machines.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    # Load exactly the trained fusion architecture and checkpoint weights.
    model = RGBThermalFusionNet(num_classes=NUM_CLASSES)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Build transforms from args to match selected input resolution.
    rgb_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    thm_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Convert images into model tensors with batch dimension.
    rgb = rgb_tf(Image.open(args.rgb).convert("RGB")).unsqueeze(0).to(device)
    thm = thm_tf(Image.open(args.thermal).convert("L")).unsqueeze(0).to(device)

    with torch.no_grad():
        _, probs = model(rgb, thm)
    idx = int(probs[0].argmax())
    print(CLASS_NAMES[idx], float(probs[0][idx]))


if __name__ == "__main__":
    main()
