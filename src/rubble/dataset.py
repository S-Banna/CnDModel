import os
import json
import random
from PIL import Image

import torch
import numpy as np


class RubbleDataset:
    def __init__(self, data_root, subset="tier1"):
        self.base = os.path.join(data_root, subset)

        self.images_dir = os.path.join(self.base, "images")
        self.masks_dir = os.path.join(self.base, "masks")
        self.labels_dir = os.path.join(self.base, "labels")

        self.samples = []

        for file in os.listdir(self.images_dir):
            if file.endswith("_pre_disaster.png"):
                base_name = file.replace("_pre_disaster.png", "")

                pre = os.path.join(
                    self.images_dir,
                    f"{base_name}_pre_disaster.png"
                )

                post = os.path.join(
                    self.images_dir,
                    f"{base_name}_post_disaster.png"
                )

                mask = os.path.join(
                    self.masks_dir,
                    f"{base_name}_post_disaster.png"
                )

                label = os.path.join(
                    self.labels_dir,
                    f"{base_name}_post_disaster.json"
                )

                if (
                    os.path.exists(pre)
                    and os.path.exists(post)
                    and os.path.exists(mask)
                    and os.path.exists(label)
                ):
                    self.samples.append({
                        "name": base_name,
                        "pre": pre,
                        "post": post,
                        "mask": mask,
                        "label": label,
                    })

    def __len__(self):
        return len(self.samples)

    def get_random(self):
        return random.choice(self.samples)

    def get_by_name(self, name):
        for s in self.samples:
            if s["name"] == name:
                return s
        return None


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    return img


def load_mask(path):
    mask = Image.open(path).convert("L")
    mask = np.array(mask).astype(np.float32)

    if mask.max() > 1:
        mask = mask / 255.0

    return mask


def load_gsd(label_path):
    with open(label_path, "r") as f:
        data = json.load(f)

    return data["metadata"]["gsd"]