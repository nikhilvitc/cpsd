# MODULE USAGE:
# - argparse: Parse command-line arguments (--kaist-root, --epochs, --batch-size, --img-size, etc.).
# - torch: Deep learning operations, device management, and checkpoint saving.
# - prepare_kaist_dataset: Load paired RGB/thermal images with binary pedestrian labels.
# - rgbt_fusion_model: Define fusion architecture and training loop.

import argparse

import torch

from prepare_kaist_dataset import NUM_CLASSES, build_kaist_loaders
from rgbt_fusion_model import RGBThermalFusionNet, train_model


def parse_args():
    p = argparse.ArgumentParser()
    # KAIST root should contain images/ and annotation folders.
    p.add_argument("--kaist-root", required=True)
    # More epochs usually improves fit, but increases training time.
    p.add_argument("--epochs", type=int, default=10)
    # Batch size controls memory usage vs throughput.
    p.add_argument("--batch-size", type=int, default=16)
    # Input resize for both RGB and thermal streams.
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--checkpoint", default="best_model_kaist.pth")
    return p.parse_args()


def main():
    args = parse_args()
    # Prefer CUDA, then Apple MPS, then CPU so the same script runs everywhere.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Build aligned RGB-thermal loaders with the same split/resize policy.
    tr_loader, va_loader, _ = build_kaist_loaders(
        kaist_root=args.kaist_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        img_size=args.img_size,
    )
    # Keep architecture fixed to the project fusion model.
    model = RGBThermalFusionNet(num_classes=NUM_CLASSES)
    train_model(model, tr_loader, va_loader, args.epochs, args.lr, device, args.checkpoint)


if __name__ == "__main__":
    main()
