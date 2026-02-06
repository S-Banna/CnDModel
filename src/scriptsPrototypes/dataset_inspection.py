import os
import json
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shapely import wkt
from shapely.geometry import mapping
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

# -----------------------------
# LOAD CONFIG
# -----------------------------
def load_config():
    config_path = "../../data/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["DATA_ROOT"]

DATA_ROOT = load_config()


# -----------------------------
# 1. CLASS DISTRIBUTION CHECK
# -----------------------------
def compute_class_distribution(data_root):
    class_counts = {}

    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith("_post_disaster.json"):
                json_path = os.path.join(root, file)

                with open(json_path) as f:
                    data = json.load(f)

                features = data["features"]["lng_lat"]

                for feat in features:
                    subtype = feat["properties"]["subtype"]
                    class_counts[subtype] = class_counts.get(subtype, 0) + 1

    print("\nBuilding subtype distribution:")
    for k, v in class_counts.items():
        print(f"{k}: {v}")

    return class_counts


# -----------------------------
# 2. LOAD ONE SAMPLE
# -----------------------------
def load_random_sample(data_root):
    post_json_files = []

    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith("_post_disaster.json"):
                post_json_files.append(os.path.join(root, file))

    chosen_json = random.choice(post_json_files)

    with open(chosen_json) as f:
        post_data = json.load(f)

    labels_dir = os.path.dirname(chosen_json)
    images_dir = os.path.join(os.path.dirname(labels_dir), "images")

    img_name = post_data["metadata"]["img_name"]

    # Force .tif extension
    img_name = img_name.replace(".png", ".tif")

    post_img_path = os.path.join(images_dir, img_name)
    pre_img_path = os.path.join(images_dir, img_name.replace("_post_", "_pre_"))

    with rasterio.open(pre_img_path) as src:
        pre_img = src.read().transpose(1, 2, 0)

    with rasterio.open(post_img_path) as src:
        post_img = src.read().transpose(1, 2, 0)

    return pre_img, post_img, post_data


# -----------------------------
# 3. RASTERIZE DAMAGE MASK
# -----------------------------
def rasterize_damage_mask(post_data, height, width):
    polygons = []

    for feat in post_data["features"]["lng_lat"]:
        geom = wkt.loads(feat["wkt"])
        subtype = feat["properties"]["subtype"]

        # Binary collapse
        value = 1 if subtype in ["major-damage", "destroyed"] else 0
        polygons.append((geom, value))

    if not polygons:
        # return empty mask if no buildings
        return np.zeros((height, width), dtype=np.uint8)

    bounds = (
        min([geom.bounds[0] for geom, _ in polygons]),
        min([geom.bounds[1] for geom, _ in polygons]),
        max([geom.bounds[2] for geom, _ in polygons]),
        max([geom.bounds[3] for geom, _ in polygons]),
    )

    transform = from_bounds(*bounds, width=width, height=height)

    mask = rasterize(
        [(mapping(geom), value) for geom, value in polygons],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    return mask


# -----------------------------
# 4. VISUALIZATION
# -----------------------------
def visualize(pre_img, post_img, mask):
    overlay = post_img.copy()
    overlay[mask == 1] = [255, 0, 0]

    fig, axs = plt.subplots(1, 4, figsize=(18, 5))

    axs[0].imshow(pre_img)
    axs[0].set_title("Pre Image")
    axs[0].axis("off")

    axs[1].imshow(post_img)
    axs[1].set_title("Post Image")
    axs[1].axis("off")

    axs[2].imshow(mask, cmap="gray")
    axs[2].set_title("Binary Damage Mask")
    axs[2].axis("off")

    axs[3].imshow(overlay)
    axs[3].set_title("Overlay (Red = Damaged)")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print(f"\nUsing DATA_ROOT: {DATA_ROOT}")

    compute_class_distribution(DATA_ROOT)

    pre_img, post_img, post_data = load_random_sample(DATA_ROOT)

    h, w, _ = post_img.shape
    mask = rasterize_damage_mask(post_data, h, w)

    visualize(pre_img, post_img, mask)