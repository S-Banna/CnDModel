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

class XView2Dataset(Dataset):
    def __init__(self, images_dir, targets_dir):
        self.images_dir = images_dir
        self.targets_dir = targets_dir

        # Only use post images to define pairs
        self.post_images = [
            f for f in os.listdir(images_dir)
            if "_post" in f
        ]

    def __len__(self):
        return len(self.post_images)

    def __getitem__(self, idx):
        post_fname = self.post_images[idx]
        pre_fname = post_fname.replace("_post", "_pre")

        # ---- Load images ----
        pre_img = np.array(Image.open(os.path.join(self.images_dir, pre_fname)))
        post_img = np.array(Image.open(os.path.join(self.images_dir, post_fname)))

        # Normalize to [0,1] and convert to float32
        pre_img = pre_img.astype(np.float32) / 255.0
        post_img = post_img.astype(np.float32) / 255.0

        # Stack into 6-channel image
        stacked = np.concatenate([pre_img, post_img], axis=2)  # (H,W,6)

        # Convert to channel-first tensor (6,H,W)
        image_tensor = torch.from_numpy(stacked).permute(2, 0, 1)

        # ---- Load mask ----
        mask_fname = post_fname.replace(".png", "_target.png")
        mask = np.array(Image.open(os.path.join(self.targets_dir, mask_fname)))

        # Convert mask to binary (major/destroyed > 2 for this dataset)
        # xView2 encoding:
        # 1=no-damage
        # 2=minor
        # 3=major
        # 4=destroyed
        binary_mask = np.isin(mask, [3, 4]).astype(np.float32)

        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)  # (1,H,W)

        return image_tensor, mask_tensor


# -------------------------
# DATALOADER
# -------------------------

dataset = XView2Dataset(IMAGES_DIR, TARGETS_DIR)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

print("Dataset size:", len(dataset))


# -------------------------
# VISUALIZE ONE BATCH
# -------------------------

images, masks = next(iter(loader))

print("Batch image shape:", images.shape)  # (B,6,H,W)
print("Batch mask shape:", masks.shape)    # (B,1,H,W)

print(masks.unique())
print(images.min(), images.max())


# Take first sample in batch
img = images[0]
mask = masks[0]

# Split back to pre/post for visualization
pre = img[:3].permute(1, 2, 0).numpy()
post = img[3:].permute(1, 2, 0).numpy()
mask_np = mask.squeeze().numpy()

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(pre)
axes[0].set_title("Pre")
axes[0].axis("off")

axes[1].imshow(post)
axes[1].set_title("Post")
axes[1].axis("off")

axes[2].imshow(post)
axes[2].imshow(mask_np, alpha=0.4)
axes[2].set_title("Post + Binary Mask")
axes[2].axis("off")

axes[3].imshow(mask_np, cmap="gray")
axes[3].set_title("Binary Mask")
axes[3].axis("off")

plt.tight_layout()
plt.show()
