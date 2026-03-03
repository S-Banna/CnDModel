import torch
import torch.nn as nn
import os, yaml
from dataset import XVDataset
from torch.utils.data import DataLoader
from unet import UNet

# ---- CONFIG ----
def load_config():
    with open("../../data/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config["DATA_ROOT"]

DATA_ROOT = load_config()
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
TARGETS_DIR = os.path.join(DATA_ROOT, "targets")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Data ----
dataset = XVDataset(IMAGES_DIR, TARGETS_DIR)
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

images, masks = next(iter(loader))

images = images.to(device)
masks = masks.to(device)

# ---- Model ----
model = UNet().to(device)

# ---- Loss ----
criterion = nn.BCEWithLogitsLoss()

# ---- Forward ----
outputs = model(images)

print("Output shape:", outputs.shape)

# ---- Compute Loss ----
loss = criterion(outputs, masks)
print("Loss:", loss.item())

# ---- Backward ----
loss.backward()

# ---- Check gradients ----
print("Grad mean:",
      model.enc1[0].weight.grad.abs().mean().item())