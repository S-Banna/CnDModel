import os
import yaml

import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider
from matplotlib.widgets import RadioButtons

import segmentation_models_pytorch as smp

from dataset import (
    RubbleDataset,
    load_image,
    load_mask,
    load_gsd
)

from utils import (
    clean_prediction,
    extract_buildings
)

from quantify import quantify_building


# -------------------------
# CONFIG
# -------------------------

def load_config():
    with open("../../data/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config["DATA_ROOT"]


DATA_ROOT = load_config()

dataset = RubbleDataset(DATA_ROOT, subset="tier1")

sample = dataset.get_random()

print(sample["name"])


# -------------------------
# LOAD DATA
# -------------------------

pre = load_image(sample["pre"])
post = load_image(sample["post"])
truth = load_mask(sample["mask"])

gsd = load_gsd(sample["label"])

stacked = np.concatenate([pre, post], axis=2)

tensor = (
    torch.tensor(stacked)
    .permute(2, 0, 1)
    .unsqueeze(0)
    .float()
)


# -------------------------
# MODEL
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=6,
    classes=1,
).to(device)

model.load_state_dict(
    torch.load(
        "../segmentation/model.pth",
        map_location=device
    )
)

model.eval()

with torch.no_grad():
    logits = model(tensor.to(device))

probs = (
    torch.sigmoid(logits)
    .cpu()
    .squeeze()
    .numpy()
)


# -------------------------
# INITIAL PREDICTION
# -------------------------

threshold = 0.97

binary = (probs > threshold).astype(np.uint8)

cleaned = clean_prediction(binary)

buildings = extract_buildings(cleaned)

structure_type = "Residential Low Rise"


# -------------------------
# FIGURE
# -------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plt.subplots_adjust(
    left=0.08,
    right=0.5,
    bottom=0.20
)

axes[0, 0].imshow(pre)
axes[0, 0].set_title("Pre")

axes[0, 1].imshow(post)
axes[0, 1].set_title("Post")

axes[1, 0].imshow(truth, cmap="gray")
axes[1, 0].set_title("Ground Truth")

pred_display = axes[1, 1].imshow(
    cleaned,
    cmap="gray"
)

axes[1, 1].set_title("Prediction")

for ax in axes.flatten():
    ax.axis("off")


# -------------------------
# TABLE
# -------------------------

table_ax = plt.axes([0.55, 0.25, 0.4, 0.55])
table_ax.axis("off")


def build_table():
    table_ax.clear()
    table_ax.axis("off")

    if not buildings:
        table_ax.text(
            0.5, 0.5, 
            "No buildings detected\nwith current threshold.", 
            ha="center", va="center", 
            color="red", fontsize=10, weight="bold"
        )
        return

    MAX_ROWS = 15

    rows = []

    for b in buildings[:MAX_ROWS]:
        q = quantify_building(
            b,
            gsd,
            structure_type
        )

        rows.append([
            q["building_id"],
            int(q["area_m2"]),
            int(q["volume_m3"]),
            int(q["concrete_kg"]),
            int(q["steel_kg"]),
            int(q["masonry_kg"]),
            int(q["mass_kg"])
        ])

    if len(buildings) > MAX_ROWS:
        rows.append(["...", f"+{len(buildings) - MAX_ROWS} more", "", "", "", "", ""])

    table = table_ax.table(
        cellText=rows,
        colLabels=[
            "ID",
            "m²",
            "m³",
            "Concrete kg",
            "Steel kg",
            "Masonry kg",
            "Total kg"
        ],
        colWidths=[
            0.08,
            0.12,
            0.12,
            0.20,
            0.16,
            0.16,
            0.16
        ],
        loc="center"
    )

    

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)


build_table()


# -------------------------
# THRESHOLD SLIDER
# -------------------------

slider_ax = plt.axes([0.15, 0.08, 0.45, 0.03])

slider = Slider(
    slider_ax,
    "Threshold",
    0.0,
    1.0,
    valinit=threshold
)


# -------------------------
# STRUCTURE SELECT
# -------------------------

radio_ax = plt.axes([0.78, 0.05, 0.18, 0.15])

radio = RadioButtons(
    radio_ax,
    (
        "Residential Low Rise",
        "Residential High Rise",
        "Industrial"
    )
)


# -------------------------
# UPDATE
# -------------------------

def update(val):
    global buildings
    global structure_type

    thresh = slider.val

    structure_type = radio.value_selected

    binary = (probs > thresh).astype(np.uint8)

    cleaned = clean_prediction(binary)

    buildings = extract_buildings(cleaned)

    pred_display.set_data(cleaned)

    build_table()

    fig.canvas.draw_idle()


slider.on_changed(update)

radio.on_clicked(update)


# -------------------------
# SHOW
# -------------------------

plt.show()