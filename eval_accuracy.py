import torch
from prepare_kaist_dataset import build_kaist_loaders, NUM_CLASSES
from rgbt_fusion_model import RGBThermalFusionNet

root = '/Users/nikhilkumar/Downloads/kaist_preview/extracted/kaist-cvpr15-preview'
_, val_loader, _ = build_kaist_loaders(kaist_root=root, batch_size=32, val_split=0.2)

device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
model = RGBThermalFusionNet(num_classes=NUM_CLASSES)
ckpt = torch.load('best_model_kaist.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

print("Evaluating model on validation set...")
correct = 0
total = 0
with torch.no_grad():
    for rgb, thm, y in val_loader:
        rgb = rgb.to(device)
        thm = thm.to(device)
        _, probs = model(rgb, thm)
        preds = probs.argmax(dim=1)
        correct += (preds == y.to(device)).sum().item()
        total += y.size(0)

accuracy = 100.0 * correct / total
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
