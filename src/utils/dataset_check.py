import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import yaml

# ------------------------
# CONFIG
# ------------------------
def load_config():
    config_path = "../../data/config.yaml"  # adjust path if needed
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["DATA_ROOT"]

DATA_ROOT = load_config()
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
TARGETS_DIR = os.path.join(DATA_ROOT, "targets")

# ------------------------
# HELPERS
# ------------------------
def get_post_images():
    """Return list of post-disaster image filenames."""
    return [f for f in os.listdir(IMAGES_DIR) if "_post" in f]

def load_pair(fname):
    """Load pre/post images and the target mask."""
    pre_fname = fname.replace("_post", "_pre")
    pre_img = np.array(Image.open(os.path.join(IMAGES_DIR, pre_fname)))
    post_img = np.array(Image.open(os.path.join(IMAGES_DIR, fname)))
    mask_fname = fname.replace(".png", "_target.png")
    mask = np.array(Image.open(os.path.join(TARGETS_DIR, mask_fname)))
    
    return pre_img, post_img, mask

def visualize(pre_img, post_img, mask):
    """Show pre/post images and overlay mask."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(pre_img)
    axes[0].set_title("Pre Image")
    axes[0].axis("off")
    
    axes[1].imshow(post_img)
    axes[1].imshow(mask, alpha=0.4)
    axes[1].set_title("Post + Mask Overlay")
    axes[1].axis("off")
    
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Binary Damage Mask")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()

# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    post_images = get_post_images()
    
    for _ in range(5):
        fname = random.choice(post_images)
        pre, post, mask = load_pair(fname)
        print(fname)
        visualize(pre, post, mask)
