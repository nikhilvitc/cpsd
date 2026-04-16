"""KAIST loader for binary pedestrian classification (no_pedestrian, pedestrian)."""

# MODULE USAGE:
# - argparse: (Not used directly; CLI provided by train_kaist.py)
# - PIL.Image: Load RGB and thermal image files from KAIST dataset tree.
# - torch / torch.utils.data: Create DataLoader and Dataset wrappers for batching.
# - torchvision.transforms: Normalize RGB (ImageNet) and thermal (single-channel) images.
# - xml.etree.ElementTree: Parse XML annotation files for label extraction.
# - os / pathlib: Navigate KAIST directory hierarchy (images/, annotation/).

import os
import random
import xml.etree.ElementTree as ET
from collections import Counter
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASS_NAMES = ["no_pedestrian", "pedestrian"]
NUM_CLASSES = len(CLASS_NAMES)
# Tokens treated as pedestrian-positive in KAIST annotations.
PEDESTRIAN_TOKENS = {"person", "people", "cyclist"}


class KAISTPairedDataset(Dataset):
    def __init__(self, rgb_paths: List[str], thm_paths: List[str], labels: List[int], img_size: int = 224):
        self.rgb_paths = rgb_paths
        self.thm_paths = thm_paths
        self.labels = labels
        # Keep RGB preprocessing close to ImageNet-style statistics.
        self.rgb_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Thermal images carry intensity only; grayscale + 1-channel normalization is sufficient.
        self.thm_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
        thm = Image.open(self.thm_paths[idx]).convert("L")
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return self.rgb_tf(rgb), self.thm_tf(thm), y


def _has_ped_txt(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return False
    for line in lines:
        t = line.strip()
        if not t or t.startswith("%"):
            continue
        # Class name is the first token in KAIST txt rows.
        if t.split()[0].lower() in PEDESTRIAN_TOKENS:
            return True
    return False


def _has_ped_xml(path: str) -> bool:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return False
    for obj in root.findall("object"):
        name = obj.find("name")
        if name is not None and name.text and name.text.strip().lower() in PEDESTRIAN_TOKENS:
            return True
    return False


def _collect_pairs(root: str) -> Tuple[List[str], List[str], List[int]]:
    images_root = os.path.join(root, "images")
    txt_root = os.path.join(root, "annotations")
    xml_root = os.path.join(root, "annotations-xml-new-sanitized")

    rgb_paths, thm_paths, labels = [], [], []
    # Walk full KAIST hierarchy: images/setXX/VXXX/{visible,lwir}
    for set_name in sorted(os.listdir(images_root)):
        set_dir = os.path.join(images_root, set_name)
        if not os.path.isdir(set_dir):
            continue
        for vid in sorted(os.listdir(set_dir)):
            vid_dir = os.path.join(set_dir, vid)
            vdir = os.path.join(vid_dir, "visible")
            ldir = os.path.join(vid_dir, "lwir")
            if not (os.path.isdir(vdir) and os.path.isdir(ldir)):
                continue

            ann_txt_dir = os.path.join(txt_root, set_name, vid)
            ann_xml_dir = os.path.join(xml_root, set_name, vid)

            for name in sorted(os.listdir(vdir)):
                if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                rgb_fp = os.path.join(vdir, name)
                thm_fp = os.path.join(ldir, name)
                stem = os.path.splitext(name)[0]
                txt_fp = os.path.join(ann_txt_dir, stem + ".txt")
                xml_fp = os.path.join(ann_xml_dir, stem + ".xml")
                if not (os.path.isfile(rgb_fp) and os.path.isfile(thm_fp)):
                    continue
                # Prefer txt annotations when available; fallback to xml for preview variants.
                if os.path.isfile(txt_fp):
                    label = 1 if _has_ped_txt(txt_fp) else 0
                elif os.path.isfile(xml_fp):
                    label = 1 if _has_ped_xml(xml_fp) else 0
                else:
                    continue
                rgb_paths.append(rgb_fp)
                thm_paths.append(thm_fp)
                labels.append(label)
    return rgb_paths, thm_paths, labels


def build_kaist_loaders(
    kaist_root: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    img_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
):
    rgb, thm, y = _collect_pairs(kaist_root)
    n = len(y)
    if n == 0:
        raise RuntimeError("No valid KAIST pairs found")

    dist = Counter(y)
    print(f"Total paired samples loaded : {n}")
    print(f"  class 0 (no_pedestrian): {dist[0]}")
    print(f"  class 1 (pedestrian)   : {dist[1]}")

    # Deterministic shuffle for reproducible train/val split.
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    split = int(n * (1 - val_split))
    tr, va = idx[:split], idx[split:]

    def sub(ids):
        return [rgb[i] for i in ids], [thm[i] for i in ids], [y[i] for i in ids]

    tr_rgb, tr_thm, tr_y = sub(tr)
    va_rgb, va_thm, va_y = sub(va)

    tr_ds = KAISTPairedDataset(tr_rgb, tr_thm, tr_y, img_size)
    va_ds = KAISTPairedDataset(va_rgb, va_thm, va_y, img_size)

    # Shuffle training only; validation order does not matter for accuracy metrics.
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return tr_loader, va_loader, CLASS_NAMES
