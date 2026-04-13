import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from prepare_kaist_dataset import CLASS_NAMES, NUM_CLASSES
from rgbt_fusion_model import RGBThermalFusionNet


RGB_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
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
    p.add_argument("--checkpoint", default="best_model_kaist.pth")
    args = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    model = RGBThermalFusionNet(num_classes=NUM_CLASSES)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    rgb = RGB_TF(Image.open(args.rgb).convert("RGB")).unsqueeze(0).to(device)
    thm = THM_TF(Image.open(args.thermal).convert("L")).unsqueeze(0).to(device)

    with torch.no_grad():
        _, probs = model(rgb, thm)
    idx = int(probs[0].argmax())
    print(CLASS_NAMES[idx], float(probs[0][idx]))


if __name__ == "__main__":
    main()
