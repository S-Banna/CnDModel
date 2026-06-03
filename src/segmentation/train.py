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
    # hold subset -> split into validation + testing
    hold_dataset = XVDataset(DATA_ROOT, subsets=["hold"], crop_size=256, damage_only=True)

    split_file = "split_indices.pt"

    if os.path.exists(split_file):

        split_data = torch.load(split_file)

        val_indices = split_data["val_indices"]
        test_indices = split_data["test_indices"]

        val_dataset = torch.utils.data.Subset(hold_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(hold_dataset, test_indices)

        print("Loaded existing split.")

    else:

        val_size = int(0.5 * len(hold_dataset))
        test_size = len(hold_dataset) - val_size

        val_dataset, test_dataset = random_split(
            hold_dataset,
            [val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        val_indices = val_dataset.indices
        test_indices = test_dataset.indices

        torch.save({
            "val_indices": val_indices,
            "test_indices": test_indices,
            "val_size": val_size,
            "test_size": test_size,
        }, split_file)

        print("Created and saved new split.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

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
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_iou": best_iou,
                "epoch": epoch + 1,
            }, "best_model.pt")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f} Dice: {avg_dice:.4f} Mean: {avg_mean:.4f}) - Val IoU: {val_iou:.4f} - LR: {current_lr:.2e}")

    print("\nRunning final test evaluation...")

    # load best saved model
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # Initialize all metrics
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    threshold = 0.5
    test_iou_sum = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            
            # Calculate per-image IoU (matching training script)
            test_iou_sum += compute_iou(outputs, masks)
            
            # Calculate confusion matrix (matching accuracy script)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
            TP += ((preds == 1) & (masks == 1)).sum().item()
            FP += ((preds == 1) & (masks == 0)).sum().item()
            TN += ((preds == 0) & (masks == 0)).sum().item()
            FN += ((preds == 0) & (masks == 1)).sum().item()

    # Calculate metrics (same as accuracy script)
    test_iou = test_iou_sum / len(test_loader)
    
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    global_iou = TP / (TP + FP + FN + 1e-6)
    oa = (TP + TN) / (TP + TN + FP + FN + 1e-6)

    print(f"\n=== FINAL TEST METRICS ===")
    print(f"Per-image Avg IoU (training style): {test_iou:.4f}")
    print(f"Global IoU (confusion matrix style): {global_iou:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Overall Accuracy (OA): {oa:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"TP: {TP:,}")
    print(f"FP: {FP:,}")
    print(f"TN: {TN:,}")
    print(f"FN: {FN:,}")

    # Save metrics to file for later reference
    metrics = {
        'test_iou_per_image': test_iou,
        'test_iou_global': global_iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'oa': oa,
        'tp': TP,
        'fp': FP,
        'tn': TN,
        'fn': FN
    }
    torch.save(metrics, "test_metrics.pt")
    print(f"\nMetrics saved to test_metrics.pt")

if __name__ == "__main__":
    main()