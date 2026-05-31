import os
import json
import random
import numpy as np
from PIL import Image


def pick_sample(data_root, subset="tier1"):
    images_dir = os.path.join(data_root, subset, "images")
    masks_dir  = os.path.join(data_root, subset, "masks")
    labels_dir = os.path.join(data_root, subset, "labels")

    candidates = [
        f.replace("_pre_disaster.png", "")
        for f in os.listdir(images_dir)
        if f.endswith("_pre_disaster.png")
    ]

    random.shuffle(candidates)

    for name in candidates:
        paths = {
            "name":  name,
            "pre":   os.path.join(images_dir, f"{name}_pre_disaster.png"),
            "post":  os.path.join(images_dir, f"{name}_post_disaster.png"),
            "mask":  os.path.join(masks_dir,  f"{name}_post_disaster.png"),
            "label": os.path.join(labels_dir, f"{name}_post_disaster.json"),
        }
        if all(os.path.exists(p) for p in paths.values() if isinstance(p, str) and p != paths["name"]):
            return paths

    raise FileNotFoundError("No complete sample found in dataset.")


def load_image(path):
    return np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0

def load_mask(path):
    mask = np.array(Image.open(path).convert("L")).astype(np.float32)
    return mask / 255.0 if mask.max() > 1 else mask

def load_gsd(label_path):
    with open(label_path) as f:
        meta = json.load(f)["metadata"]
    if "pan_resolution" in meta:
        return meta["pan_resolution"]
    return meta["gsd"] / 4.0