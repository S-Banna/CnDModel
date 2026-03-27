import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


# -------------------------
# CONFIG
# -------------------------

def load_config():
    with open("../../data/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config["DATA_ROOT"]

DATA_ROOT = load_config()
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
TARGETS_DIR = os.path.join(DATA_ROOT, "targets")


# -------------------------
# DATASET
# -------------------------
class XVDataset(Dataset):
    def __init__(self, images_dir, targets_dir, crop_size=256, damage_only=False):
        self.images_dir = images_dir
        self.targets_dir = targets_dir
        self.crop_size = crop_size

        self.post_images = [
            f for f in os.listdir(images_dir)
            if "_post" in f
        ]

        if damage_only:
            self.post_images = [
                f for f in self.post_images
                if np.isin(
                    np.array(Image.open(os.path.join(targets_dir, f.replace(".png", "_target.png")))),
                    [3, 4]
                ).any()
            ]

    def __len__(self):
        return len(self.post_images)

    def __getitem__(self, idx):

        post_fname = self.post_images[idx]
        pre_fname = post_fname.replace("_post", "_pre")

        # ---- Load images ----
        pre_img = np.array(Image.open(os.path.join(self.images_dir, pre_fname)))
        post_img = np.array(Image.open(os.path.join(self.images_dir, post_fname)))

        pre_img = pre_img.astype(np.float32) / 255.0
        post_img = post_img.astype(np.float32) / 255.0

        stacked = np.concatenate([pre_img, post_img], axis=2)  # (H,W,6)

        # ---- Load mask ----
        mask_fname = post_fname.replace(".png", "_target.png")
        mask = np.array(Image.open(os.path.join(self.targets_dir, mask_fname)))

        binary_mask = np.isin(mask, [3,4]).astype(np.float32)

        # ----------------------
        # RANDOM CROP + BIAS
        # ----------------------

        H, W, _ = stacked.shape
        cs = self.crop_size

        damage_indices = np.argwhere(binary_mask == 1)

        use_damage_crop = len(damage_indices) > 0 and random.random() < 0.5

        if use_damage_crop:
            # pick a random damage pixel
            y_center, x_center = damage_indices[random.randint(0, len(damage_indices) - 1)]

            # convert center → top-left
            y = y_center - cs // 2
            x = x_center - cs // 2

            # clamp to image bounds
            y = max(0, min(y, H - cs))
            x = max(0, min(x, W - cs))

        else:
            # fallback: random crop
            y = random.randint(0, H - cs)
            x = random.randint(0, W - cs)

        stacked = stacked[y:y+cs, x:x+cs]
        binary_mask = binary_mask[y:y+cs, x:x+cs]

        # ----------------------

        image_tensor = torch.from_numpy(stacked).permute(2,0,1)
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)

        return image_tensor, mask_tensor