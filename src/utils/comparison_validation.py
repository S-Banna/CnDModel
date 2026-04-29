"""
Compares legacy dataset against tier1 of the new xbd dataset.
Accounts for:
  - legacy targets: ..._post_disaster_target.png
  - tier1 masks:    ..._post_disaster.png  (no _target suffix)
Run from src/model/ as usual.
"""

import os
import yaml
import numpy as np
from PIL import Image

CONFIG_PATH = "../../data/config.yaml"
MANUAL_PIXEL_CHECK_COUNT = 5

# -------------------------
# LOAD CONFIG
# -------------------------
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

LEGACY_ROOT = config["LEGACY_DATA_ROOT"]
NEW_ROOT    = config["DATA_ROOT"]

LEGACY_IMAGES  = os.path.join(LEGACY_ROOT, "images")
LEGACY_TARGETS = os.path.join(LEGACY_ROOT, "targets")
TIER1_IMAGES   = os.path.join(NEW_ROOT, "tier1", "images")
TIER1_MASKS    = os.path.join(NEW_ROOT, "tier1", "masks")

for path, label in [
    (LEGACY_IMAGES,  "legacy images"),
    (LEGACY_TARGETS, "legacy targets"),
    (TIER1_IMAGES,   "tier1 images"),
    (TIER1_MASKS,    "tier1 masks"),
]:
    if not os.path.exists(path):
        print(f"❌ Path not found: {path} ({label})")
        exit(1)

# -------------------------
# COLLECT FILES
# -------------------------
def list_pngs(folder):
    return set(f for f in os.listdir(folder) if f.endswith(".png"))

legacy_images   = list_pngs(LEGACY_IMAGES)
legacy_targets  = list_pngs(LEGACY_TARGETS)
tier1_images    = list_pngs(TIER1_IMAGES)
tier1_masks_all = list_pngs(TIER1_MASKS)

# filter out _rgb variants
tier1_masks = set(f for f in tier1_masks_all if "_rgb" not in f)

# normalize legacy target names to match tier1 mask names (strip _target)
# e.g. guatemala-volcano_00000000_post_disaster_target.png
#   -> guatemala-volcano_00000000_post_disaster.png
legacy_targets_normalized = {
    f.replace("_target.png", ".png"): f
    for f in legacy_targets
}

# -------------------------
# FILE COUNT
# -------------------------
print("=" * 60)
print("FILE COUNT COMPARISON")
print("=" * 60)
print(f"  Legacy images          : {len(legacy_images)}")
print(f"  Tier1  images          : {len(tier1_images)}")
print(f"  Legacy targets         : {len(legacy_targets)}")
print(f"  Tier1  masks (excl rgb): {len(tier1_masks)}")
print(f"  Tier1  _rgb masks      : {len(tier1_masks_all) - len(tier1_masks)}")

# -------------------------
# IMAGE MATCHING
# -------------------------
print("\n" + "=" * 60)
print("IMAGE FILENAME MATCHING")
print("=" * 60)
in_both_img             = legacy_images & tier1_images
in_legacy_not_tier1_img = legacy_images - tier1_images
in_tier1_not_legacy_img = tier1_images  - legacy_images

print(f"  In both            : {len(in_both_img)}")
print(f"  Legacy only        : {len(in_legacy_not_tier1_img)}")
print(f"  Tier1 only         : {len(in_tier1_not_legacy_img)}")
if in_legacy_not_tier1_img:
    print(f"  ⚠️  Legacy-only (first 5): {sorted(in_legacy_not_tier1_img)[:5]}")
if in_tier1_not_legacy_img:
    print(f"  ⚠️  Tier1-only  (first 5): {sorted(in_tier1_not_legacy_img)[:5]}")

# -------------------------
# MASK MATCHING (normalized names)
# -------------------------
print("\n" + "=" * 60)
print("MASK FILENAME MATCHING (after stripping _target)")
print("=" * 60)
legacy_norm_set         = set(legacy_targets_normalized.keys())
in_both_tgt             = legacy_norm_set & tier1_masks
in_legacy_not_tier1_tgt = legacy_norm_set - tier1_masks
in_tier1_not_legacy_tgt = tier1_masks     - legacy_norm_set

print(f"  In both            : {len(in_both_tgt)}")
print(f"  Legacy only        : {len(in_legacy_not_tier1_tgt)}")
print(f"  Tier1 only         : {len(in_tier1_not_legacy_tgt)}")
if in_legacy_not_tier1_tgt:
    print(f"  ⚠️  Legacy-only (first 5): {sorted(in_legacy_not_tier1_tgt)[:5]}")
if in_tier1_not_legacy_tgt:
    print(f"  ⚠️  Tier1-only  (first 5): {sorted(in_tier1_not_legacy_tgt)[:5]}")

# -------------------------
# PIXEL EQUALITY CHECK
# -------------------------
print("\n" + "=" * 60)
print(f"PIXEL EQUALITY CHECK ({MANUAL_PIXEL_CHECK_COUNT} files)")
print("=" * 60)

shared_images = sorted(in_both_img)
step     = max(1, len(shared_images) // MANUAL_PIXEL_CHECK_COUNT)
to_check = [shared_images[i] for i in range(0, len(shared_images), step)][:MANUAL_PIXEL_CHECK_COUNT]

all_match = True
for fname in to_check:
    legacy_path = os.path.join(LEGACY_IMAGES, fname)
    tier1_path  = os.path.join(TIER1_IMAGES,  fname)
    arr_l = np.array(Image.open(legacy_path))
    arr_t = np.array(Image.open(tier1_path))
    if arr_l.shape != arr_t.shape:
        print(f"  ❌ {fname}  SHAPE MISMATCH: {arr_l.shape} vs {arr_t.shape}")
        all_match = False
    elif not np.array_equal(arr_l, arr_t):
        diff = np.abs(arr_l.astype(int) - arr_t.astype(int))
        print(f"  ⚠️  {fname}  pixel diff — max={diff.max()} mean={diff.mean():.4f}")
        all_match = False
    else:
        print(f"  ✅ {fname}  identical")

# -------------------------
# VERDICT
# -------------------------
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
img_overlap  = 100 * len(in_both_img) / len(legacy_images) if legacy_images else 0
mask_overlap = 100 * len(in_both_tgt) / len(legacy_targets) if legacy_targets else 0
print(f"  Image overlap : {img_overlap:.1f}%")
print(f"  Mask overlap  : {mask_overlap:.1f}%")
if img_overlap > 95 and mask_overlap > 95 and all_match:
    print("  ✅ Safe to use new dataset as drop-in replacement for legacy.")
elif img_overlap > 95 and mask_overlap > 95:
    print("  ⚠️  Files match by name but pixel differences found. Investigate before switching.")
else:
    print("  ⚠️  Significant mismatch. Do not discard legacy data yet.")