import torch
import torch.nn as nn
import os, yaml
from tqdm import tqdm
from dataset import XVDataset
from torch.utils.data import DataLoader, Subset
from unet import UNet


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits) 

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice 


def load_config():
    with open("../../data/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config["DATA_ROOT"]


def main():
    DATA_ROOT = load_config()
    IMAGES_DIR = os.path.join(DATA_ROOT, "images")
    TARGETS_DIR = os.path.join(DATA_ROOT, "targets")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset = XVDataset(IMAGES_DIR, TARGETS_DIR, crop_size=256)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # model
    model = UNet().to(device)

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    epochs = 2

    for epoch in range(epochs):

        total_loss = 0

        for images, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss_bce = bce(outputs, masks)
            loss_dice = dice(outputs, masks)

            loss = loss_bce + loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()