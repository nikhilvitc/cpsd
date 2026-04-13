# RGB-T Pedestrian Classification and Highlighting

## 1. Project Title
Multi-Scale RGB-T Cross-Modal Attention Network for Pedestrian Presence Classification (KAIST) with Web Demo

## 2. Abstract
This project implements a dual-modality RGB-T pipeline for pedestrian presence recognition.
The model extracts multi-scale features from RGB and thermal images, fuses them using cross-modal attention, and predicts pedestrian vs no_pedestrian.
A Gradio web app is provided to run inference and visualize a red-box highlight based on Grad-CAM.

## 3. Objectives
1. Build an RGB-T fusion model using cross-modal attention.
2. Train and evaluate on KAIST pedestrian data.
3. Provide a simple web interface for uploading RGB + thermal images.
4. Highlight the most pedestrian-relevant region in both RGB and thermal outputs.

## 4. Dataset
Dataset used: KAIST Multispectral Pedestrian Dataset (preview/full).

Supported annotation layouts:
1. TXT (`annotations`)
2. XML (`annotations-xml-new-sanitized`)

Binary labels:
1. `no_pedestrian`
2. `pedestrian` (if annotation contains person/people/cyclist)

## 5. Architecture Overview
Pipeline:
1. RGB Feature Extractor (3 ConvBlocks)
2. Thermal Feature Extractor (3 ConvBlocks)
3. Cross-Attention at three scales
4. Global average pooling and feature concatenation
5. Linear classifier (2 classes)

Important note:
- This is a classifier architecture.
- Highlighting is Grad-CAM based and gives one main region, not true multi-object detection boxes.

## 6. Important Code Snippets

### 6.1 Conv Block
From `rgbt_fusion_model.py`:

```python
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
```

### 6.2 Cross-Modal Attention
From `rgbt_fusion_model.py`:

```python
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
```

### 6.3 Full Fusion Model
From `rgbt_fusion_model.py`:

```python
class RGBThermalFusionNet(nn.Module):
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
```

### 6.4 KAIST Data Loader
From `prepare_kaist_dataset.py`:

```python
CLASS_NAMES = ["no_pedestrian", "pedestrian"]
NUM_CLASSES = len(CLASS_NAMES)
PEDESTRIAN_TOKENS = {"person", "people", "cyclist"}

class KAISTPairedDataset(Dataset):
    def __init__(self, rgb_paths, thm_paths, labels, img_size=224):
        self.rgb_paths = rgb_paths
        self.thm_paths = thm_paths
        self.labels = labels
```

Pair collection supports both TXT and XML annotations and assigns binary labels.

### 6.5 Training Entry
From `train_kaist.py`:

```python
tr_loader, va_loader, _ = build_kaist_loaders(
    kaist_root=args.kaist_root,
    batch_size=args.batch_size,
    val_split=args.val_split,
)
model = RGBThermalFusionNet(num_classes=NUM_CLASSES)
train_model(model, tr_loader, va_loader, args.epochs, args.lr, device, args.checkpoint)
```

### 6.6 Inference Script
From `predict_kaist.py`:

```python
with torch.no_grad():
    _, probs = model(rgb, thm)
idx = int(probs[0].argmax())
print(CLASS_NAMES[idx], float(probs[0][idx]))
```

### 6.7 Web App Model Loading
From `app.py`:

```python
CHECKPOINT = os.environ.get("PEDESTRIAN_CKPT", "best_model_kaist.pth")
model = RGBThermalFusionNet(num_classes=NUM_CLASSES)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
model.eval()
```

### 6.8 Grad-CAM Localization
From `app.py`:

```python
logits, _ = MODEL(rgb_t, thm_t)
logits[:, PEDESTRIAN_CLASS_IDX].sum().backward()

a = acts["v"][0]
g = grads["v"][0]
w = g.mean(dim=(1, 2), keepdim=True)
cam = (w * a).sum(dim=0)
cam = F.relu(cam)
```

### 6.9 Red Box Highlight
From `app.py`:

```python
draw.rectangle([(bx[0], bx[1]), (bx[2], bx[3])], outline=(255, 0, 0), width=5)
draw.text((bx[0] + 4, max(0, bx[1] - 16)), "pedestrian", fill=(255, 0, 0))
```

## 7. Project Structure

```text
cps project/
├── app.py
├── rgbt_fusion_model.py
├── prepare_kaist_dataset.py
├── train_kaist.py
├── predict_kaist.py
└── README.md
```

## 8. Setup

```bash
cd "/Users/nikhilkumar/Downloads/cps project"
source .venv311/bin/activate
pip install --upgrade pip
pip install torch torchvision pillow gradio scikit-learn
```

## 9. Training Command

```bash
python train_kaist.py \
  --kaist-root "/Users/nikhilkumar/Downloads/kaist_preview/extracted/kaist-cvpr15-preview" \
  --epochs 10 \
  --batch-size 16 \
  --checkpoint best_model_kaist.pth
```

## 10. Inference Command

```bash
python predict_kaist.py \
  --rgb "/path/to/visible.jpg" \
  --thermal "/path/to/lwir.jpg" \
  --checkpoint best_model_kaist.pth
```

## 11. Run Web App

```bash
cd "/Users/nikhilkumar/Downloads/cps project"
source .venv311/bin/activate
PEDESTRIAN_CKPT="best_model_kaist.pth" python app.py
```

Default threshold can be changed:

```bash
PEDESTRIAN_THRESHOLD=0.65 python app.py
```

## 12. Results Summary Template (for Faculty Report)
Use this section in your report:

1. Dataset split used: ____
2. Training epochs: ____
3. Final validation accuracy: ____
4. Inference examples tested: ____
5. Observations:
   - strong detection cases: ____
   - failure cases: ____
   - thermal vs RGB behavior: ____

## 13. Limitations
1. Model is classifier-based, not full object detector.
2. Grad-CAM localization is approximate.
3. Multiple pedestrians may not be boxed separately.

## 14. Future Work
1. Convert to full detection architecture for all-person boxes.
2. Fine-tune with stronger KAIST training schedule.
3. Add quantitative metrics (precision/recall/F1, miss rate).
4. Add video tracking and temporal smoothing.
