import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from dataset import XVDataset
import segmentation_models_pytorch as smp
import os, yaml

# -------------------------
# CONFIG
# -------------------------
def load_config():
    with open("../../data/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config["DATA_ROOT"]

DATA_ROOT = load_config()
dataset = XVDataset(DATA_ROOT, subsets=["tier1"], crop_size=256, damage_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD MODEL
# -------------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=6,
    classes=1,
).to(device)
model.load_state_dict(torch.load("model.pth")) 
model.eval()

# -------------------------
# LOOP
# -------------------------
for i in range(len(dataset)):

    image, mask = dataset[i]

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)

    probs = torch.sigmoid(logits).cpu().squeeze().numpy()
    mask = mask.squeeze().numpy()

    # split pre/post
    stacked = image.cpu().squeeze().permute(1,2,0).numpy()
    pre = stacked[:, :, :3]
    post = stacked[:, :, 3:]

    # -------------------------
    # PLOTTING
    # -------------------------
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.25)

    axes[0].imshow(pre)
    axes[0].set_title("Pre")

    axes[1].imshow(post)
    axes[1].set_title("Post")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Ground Truth")

    pred_display = axes[3].imshow(probs > 0.05, cmap="gray", vmin=0, vmax=1) 
    axes[3].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    # -------------------------
    # SLIDER
    # -------------------------
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, "Threshold", 0.0, 1.0, valinit=0.05)

    def update(val):
        thresh = slider.val
        pred_display.set_data((probs > thresh).astype(np.float32))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

    # -------------------------
    # CONTROL
    # -------------------------
    print("probs min/max:", probs.min(), probs.max())
    cmd = input("Enter = next | q = quit: ")
    if cmd.lower() == "q":
        break