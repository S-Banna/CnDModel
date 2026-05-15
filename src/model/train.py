import torch
import torch.nn as nn
import os, yaml
from tqdm import tqdm
from dataset import XVDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import segmentation_models_pytorch as smp


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

def compute_iou(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets).clamp(0,1).sum(dim=(1,2,3))

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_iou = 0 # saves the best IoU seen at any epoch in training for visualization

    DATA_ROOT = load_config()

    # training data: tier1 + tier3
    train_dataset = XVDataset(DATA_ROOT, subsets=["tier1", "tier3"], crop_size=256, damage_only=True)

    # validation: hold (curated holdout set, never trained on)
    val_dataset = XVDataset(DATA_ROOT, subsets=["hold"], crop_size=256, damage_only=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=4, 
                              pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=4, 
                              pin_memory=True, persistent_workers=True, prefetch_factor=4)

    # model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=6,
        classes=1,
    ).to(device)

    pos_weight = torch.tensor([10.0]).to(device) 
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dice = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    model.train()

    epochs = 60

    for epoch in range(epochs):

        total_loss = 0
        total_bce  = 0
        total_dice = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
            total_bce  += loss_bce.item()
            total_dice += loss_dice.item()

        avg_loss = total_loss / len(train_loader)
        avg_bce  = total_bce  / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        avg_mean = avg_loss / 2 

        model.eval()
        val_iou = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                val_iou += compute_iou(outputs, masks)

        val_iou /= len(val_loader)
        scheduler.step(val_iou)
        model.train()

        if val_iou > best_iou: # updates the saved model if observed IoU is better than previous epochs
            best_iou = val_iou
            torch.save(model.state_dict(), "model.pth")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f} Dice: {avg_dice:.4f} Mean: {avg_mean:.4f}) - Val IoU: {val_iou:.4f} - LR: {current_lr:.2e}")


if __name__ == "__main__":
    main()