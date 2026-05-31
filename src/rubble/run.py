import yaml
import torch
import numpy as np
import segmentation_models_pytorch as smp
import cv2

from dataset import pick_sample, load_image, load_mask, load_gsd
from quantify import quantify_building


# ── config ────────────────────────────────────────────────────────────────────

with open("../../data/config.yaml") as f:
    DATA_ROOT = yaml.safe_load(f)["DATA_ROOT"]

MIN_PIXELS = 40

# Closing kernel in real-world metres — keeps blobs from merging across GSD values.
# At 0.5 m/px this is a 4px kernel; at 2.0 m/px it's a 1px kernel (effectively off).
CLOSE_METRES = 2.0

# ── data ──────────────────────────────────────────────────────────────────────

sample = pick_sample(DATA_ROOT)
print(f"Sample : {sample['name']}")

pre   = load_image(sample["pre"])
post  = load_image(sample["post"])
truth = load_mask(sample["mask"])
gsd   = load_gsd(sample["label"])

print(f"GSD    : {gsd:.4f} m/px  (pan_resolution)")
if gsd > 0.8:
    print(f"WARNING: GSD {gsd:.2f} m/px is coarser than the xBD target (<0.8 m/px). "
          f"Area estimates will be less reliable.")

# ── model ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                 in_channels=6, classes=1).to(device)
model.load_state_dict(torch.load("../segmentation/model.pth", map_location=device))
model.eval()

tensor = (torch.tensor(np.concatenate([pre, post], axis=2))
          .permute(2, 0, 1).unsqueeze(0).float())

with torch.no_grad():
    probs = torch.sigmoid(model(tensor.to(device))).cpu().squeeze().numpy()


# ── post-process (re-runs on every threshold / structure-type change) ─────────

def process(threshold, structure_type):
    binary = (probs > threshold).astype(np.uint8) * 255

    # Kernel size in pixels, physically bounded so coarse-GSD images don't merge buildings
    k = max(1, round(CLOSE_METRES / gsd))
    kernel  = np.ones((k, k), np.uint8)
    cleaned = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)

    buildings = [
        {
            "id":         i,
            "pixel_area": int(stats[i, cv2.CC_STAT_AREA]),
            "centroid":   (int(centroids[i][0]), int(centroids[i][1])),  # (x, y)
        }
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= MIN_PIXELS
    ]

    results = [quantify_building(b, gsd, structure_type) for b in buildings]

    return cleaned, buildings, results


# ── launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from viz import show
    show(pre, post, truth, probs, gsd, process)