import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml

# ------------------------
# CONFIG
# ------------------------

def load_config():
    with open("../../data/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config["DATA_ROOT"]

DATA_ROOT = load_config()

IMAGES_DIR = os.path.join(DATA_ROOT, "images")
MASKS_DIR = os.path.join(DATA_ROOT, "targets")  # precomputed masks from converter

# ------------------------
# LOAD SAMPLES
# ------------------------

def get_post_mask_files():
    return [
        f for f in os.listdir(MASKS_DIR)
        if f.endswith("_post_disaster.png")
    ]

def load_pair(mask_filename):
    """Load pre/post images and corresponding mask"""
    pre_image_name = mask_filename.replace("_post_disaster.png", "_pre_disaster.tif")
    post_image_name = mask_filename.replace("_post_disaster.png", "_post_disaster.tif")

    # PIL will handle PNG mask reading, rasterio for TIFFs
    with Image.open(os.path.join(MASKS_DIR, mask_filename)) as m:
        mask = np.array(m)

    import rasterio
    def read_tif(path):
        with rasterio.open(path) as src:
            img = src.read()
        img = np.transpose(img, (1, 2, 0))  # C,H,W -> H,W,C
        return img

    pre_img = read_tif(os.path.join(IMAGES_DIR, pre_image_name))
    post_img = read_tif(os.path.join(IMAGES_DIR, post_image_name))

    # Convert mask to binary: major+destroyed -> 1, rest -> 0
    binary_mask = np.isin(mask, [3, 4]).astype(np.uint8)

    return pre_img, post_img, binary_mask

# ------------------------
# VISUALIZATION
# ------------------------

def visualize(pre_img, post_img, mask):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1) Pre image
    axes[0].imshow(pre_img)
    axes[0].set_title("Pre Image")
    axes[0].axis("off")

    # 2) Post image
    axes[1].imshow(post_img)
    axes[1].set_title("Post Image")
    axes[1].axis("off")

    # 3) Overlay
    axes[2].imshow(post_img)
    axes[2].imshow(mask, alpha=0.4, cmap="Reds")
    axes[2].set_title("Post + Damage Mask")
    axes[2].axis("off")

    # 4) Mask on black
    axes[3].imshow(mask, cmap="gray")
    axes[3].set_title("Binary Damage Mask")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

# ------------------------
# MAIN
# ------------------------

if __name__ == "__main__":
    mask_files = get_post_mask_files()

    for i in range(5):
        mask_file = random.choice(mask_files)
        pre_img, post_img, mask = load_pair(mask_file)
        print(f"Sample {i+1}: {mask_file}")
        visualize(pre_img, post_img, mask)
