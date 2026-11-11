"""Shared, torch-free helpers for the BANDON roof/facade experiments.

Label maps (building_labels PNG): 0 = background, 1 = roof, 2 = facade.
"building" = roof U facade.
"""
import random
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = r"C:\Users\user\Downloads\BANDON"
CITIES = ("bj", "sh")            # test_ood has no roof/facade labels
DATES = ("t1", "t2", "t3")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
CLASS_NAMES = ("background", "roof", "facade")
ROOF_RGB = (255, 90, 60)
FACADE_RGB = (0, 200, 255)

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


# --------------------------------------------------------------------------
# Files / splits
# --------------------------------------------------------------------------
def split_dir(root, split):
    root = Path(root)
    return {"train": root / "train", "val": root / "val",
            "test": root / "test" / "test"}[split]


def _stem_map(folder: Path):
    if not folder.is_dir():
        return {}
    return {p.stem: p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS}


def list_tiles(root, split, n=0, seed=0):
    """All tiles with both an image and a roof/facade label. n>0 -> deterministic
    random subset (same n + seed gives the same subset for every script)."""
    base = split_dir(root, split)
    tiles = []
    for city in CITIES:
        for t in DATES:
            labs = _stem_map(base / "building_labels" / city / t)
            imgs = _stem_map(base / "imgs" / city / t)
            for stem in sorted(labs):
                if stem in imgs:
                    tiles.append({
                        "split": split, "city": city, "t": t, "stem": stem,
                        "img": str(imgs[stem]), "lab": str(labs[stem]),
                        "key": f"{city}/{stem}",   # same location across t1/t2/t3
                    })
    if n and n < len(tiles):
        tiles = sorted(random.Random(seed).sample(tiles, n),
                       key=lambda d: (d["city"], d["t"], d["stem"]))
    return tiles


def split_train_holdout(tiles, holdout_pct=10):
    """Hold out whole LOCATIONS (all dates of a location go together)."""
    train, hold = [], []
    for d in tiles:
        bucket = zlib.crc32(d["key"].encode()) % 100
        (hold if bucket < holdout_pct else train).append(d)
    return train, hold


def pred_name(tile):
    return f"{tile['city']}_{tile['t']}_{tile['stem']}.png"


# --------------------------------------------------------------------------
# Image IO
# --------------------------------------------------------------------------
def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def load_labels(path):
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.uint8)


def normalize(rgb_uint8):
    """HWC uint8 -> CHW float32 (ImageNet stats)."""
    x = (rgb_uint8.astype(np.float32) / 255.0 - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def positions(n, win, stride):
    """Window start positions covering [0, n) with a flush last window."""
    if n <= win:
        return [0]
    pos = list(range(0, n - win + 1, stride))
    if pos[-1] != n - win:
        pos.append(n - win)
    return pos


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
class ConfMat:
    def __init__(self, k=3):
        self.k = k
        self.cm = np.zeros((k, k), np.int64)    # rows = ground truth, cols = prediction

    def add(self, gt, pred):
        gt = gt.astype(np.int64).ravel()
        pred = np.clip(pred.astype(np.int64).ravel(), 0, self.k - 1)
        m = gt < self.k                          # drops ignore labels (e.g. 255)
        idx = self.k * gt[m] + pred[m]
        self.cm += np.bincount(idx, minlength=self.k ** 2).reshape(self.k, self.k)

    def scores(self):
        cm = self.cm
        tp = np.diag(cm).astype(np.float64)
        fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
        iou = tp / np.maximum(tp + fp + fn, 1)
        inter = cm[1:, 1:].sum()
        union = cm[1:, :].sum() + cm[:, 1:].sum() - inter
        building = inter / max(union, 1)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        return {
            "iou_background": float(iou[0]), "iou_roof": float(iou[1]),
            "iou_facade": float(iou[2]), "iou_building": float(building),
            "miou_roof_facade": float((iou[1] + iou[2]) / 2),
            "f1_roof": float(f1[1]), "f1_facade": float(f1[2]),
        }
