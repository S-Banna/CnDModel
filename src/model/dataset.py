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

# -------------------------
# DATASET
# -------------------------
class XVDataset(Dataset):
    def __init__(self, data_root, subsets, crop_size=256, damage_only=False):
        """
        Args:
            data_root   : path to xbd root, e.g. C:/Users/.../xbd-dataset/xbd
            subsets     : list of subset names to include, e.g. ["tier1", "tier3"]
                          each subset must have images/ and masks/ subfolders.
                          mask filenames match post image filenames exactly (no _target suffix).
            crop_size   : square crop size fed to the model
            damage_only : if True, skip images with no damage pixels (labels 3 or 4)
        """
        self.crop_size = crop_size
        # each entry: (images_dir, masks_dir, post_fname)
        self.samples = []

        for subset in subsets:
            images_dir = os.path.join(data_root, subset, "images")
            masks_dir  = os.path.join(data_root, subset, "masks")

            if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
                print(f"Skipping subset '{subset}': folder not found at {images_dir}")
                continue

            post_files = [
                f for f in os.listdir(images_dir)
                if "_post" in f and "_rgb" not in f and f.endswith(".png")
            ]

            for post_fname in post_files:
                mask_path = os.path.join(masks_dir, post_fname)
                if not os.path.exists(mask_path):
                    continue

                if damage_only:
                    mask = np.array(Image.open(mask_path))
                    if not np.isin(mask, [3, 4]).any():
                        continue

                self.samples.append((images_dir, masks_dir, post_fname))

        print(f"Dataset ready: {len(self.samples)} samples from subsets {subsets}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        images_dir, masks_dir, post_fname = self.samples[idx]
        pre_fname  = post_fname.replace("_post", "_pre")

        # ---- Load images ----
        pre_img  = np.array(Image.open(os.path.join(images_dir, pre_fname)))
        post_img = np.array(Image.open(os.path.join(images_dir, post_fname)))
        pre_img  = pre_img.astype(np.float32)  / 255.0
        post_img = post_img.astype(np.float32) / 255.0
        stacked  = np.concatenate([pre_img, post_img], axis=2)

        # ---- Load mask ----
        mask        = np.array(Image.open(os.path.join(masks_dir, post_fname)))
        binary_mask = np.isin(mask, [3, 4]).astype(np.float32)

        # ---- Random crop + damage bias ----
        H, W, _ = stacked.shape
        cs = self.crop_size
        damage_indices  = np.argwhere(binary_mask == 1)
        use_damage_crop = len(damage_indices) > 0 and random.random() < 0.5

        if use_damage_crop:
            y_center, x_center = damage_indices[random.randint(0, len(damage_indices) - 1)]
            y = max(0, min(y_center - cs // 2, H - cs))
            x = max(0, min(x_center - cs // 2, W - cs))
        else:
            y = random.randint(0, H - cs)
            x = random.randint(0, W - cs)

        stacked     = stacked[y:y+cs, x:x+cs]
        binary_mask = binary_mask[y:y+cs, x:x+cs]

        image_tensor = torch.from_numpy(stacked).permute(2, 0, 1)
        mask_tensor  = torch.from_numpy(binary_mask).unsqueeze(0)
        return image_tensor, mask_tensor