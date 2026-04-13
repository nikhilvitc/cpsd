import argparse

import torch

from prepare_kaist_dataset import NUM_CLASSES, build_kaist_loaders
from rgbt_fusion_model import RGBThermalFusionNet, train_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kaist-root", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--checkpoint", default="best_model_kaist.pth")
    return p.parse_args()


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tr_loader, va_loader, _ = build_kaist_loaders(
        kaist_root=args.kaist_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
    model = RGBThermalFusionNet(num_classes=NUM_CLASSES)
    train_model(model, tr_loader, va_loader, args.epochs, args.lr, device, args.checkpoint)


if __name__ == "__main__":
    main()
