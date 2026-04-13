"""Multi-scale RGB-T fusion classifier (without scale-aware fusion block)."""

import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class RGBThermalDataset(Dataset):
    def __init__(self, root_dir: str, filenames: List[str], labels: List[int], img_size: int = 224):
        self.root_dir = root_dir
        self.filenames = filenames
        self.labels = labels
        self.rgb_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.thm_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        name = self.filenames[idx]
        rgb = Image.open(os.path.join(self.root_dir, "rgb", name)).convert("RGB")
        thm = Image.open(os.path.join(self.root_dir, "thermal", name)).convert("L")
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return self.rgb_tf(rgb), self.thm_tf(thm), label


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RGBFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return f1, f2, f3


class ThermalFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return f1, f2, f3


class CrossAttention(nn.Module):
    def __init__(self, channels: int, attn_size: int = 14):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.pool = nn.AdaptiveAvgPool2d(attn_size)
        self.scale = channels ** -0.5
        self.attn_size = attn_size

    def forward(self, rgb_feat: torch.Tensor, thm_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = rgb_feat.shape
        m = self.attn_size * self.attn_size
        q = self.pool(self.q(rgb_feat)).view(b, c, m).transpose(1, 2)
        k = self.pool(self.k(thm_feat)).view(b, c, m)
        v = self.pool(self.v(thm_feat)).view(b, c, m).transpose(1, 2)

        attn = torch.bmm(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).view(b, c, self.attn_size, self.attn_size)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out + rgb_feat


class RGBThermalFusionNet(nn.Module):
    """Original fusion backbone: RGB/Thermal extractors + 3 cross-attention blocks."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.rgb_extractor = RGBFeatureExtractor()
        self.thm_extractor = ThermalFeatureExtractor()
        self.attn1 = CrossAttention(32)
        self.attn2 = CrossAttention(64)
        self.attn3 = CrossAttention(128)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32 + 64 + 128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor):
        f1r, f2r, f3r = self.rgb_extractor(rgb)
        f1t, f2t, f3t = self.thm_extractor(thermal)
        z1 = self.attn1(f1r, f1t)
        z2 = self.attn2(f2r, f2t)
        z3 = self.attn3(f3r, f3t)
        fused = torch.cat([
            self.gap(z1).flatten(1),
            self.gap(z2).flatten(1),
            self.gap(z3).flatten(1),
        ], dim=1)
        logits = self.classifier(fused)
        probs = self.softmax(logits)
        return logits, probs


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
    checkpoint_path: str,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_val = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        correct = 0
        total = 0
        for rgb, thm, y in train_loader:
            rgb, thm, y = rgb.to(device), thm.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(rgb, thm)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for rgb, thm, y in val_loader:
                rgb, thm, y = rgb.to(device), thm.to(device), y.to(device)
                logits, _ = model(rgb, thm)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)

        train_acc = correct / max(total, 1)
        val_acc = val_correct / max(val_total, 1)
        print(f"Epoch [{epoch}/{num_epochs}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"epoch": epoch, "val_acc": val_acc, "model_state_dict": model.state_dict()}, checkpoint_path)
