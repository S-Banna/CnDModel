"""
Per-building roof / facade segmentation with SAM 3.

Input : an image + the UNet damage mask (or a single manual box for testing)
Output: per-building roof & facade masks, full-image instance label maps, JSON

Key idea
--------
SAM 3 treats a box as a *visual exemplar* ("find more things like this"), not
as a spatial restriction, so it never "respects" the box. Instead we:
  1. split the UNet mask into islands (connected components),
  2. crop a padded square around each island (this also upsamples tiny
     buildings, which matters because SAM 3 runs at 1008x1008),
  3. run text prompts on the crop,
  4. pick instances by overlap with THAT island's mask (not just max score),
  5. make roof win over facade where they overlap.
"""
import argparse
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

DEFAULT_BPE = (
    r"C:\Users\user\Desktop\courses\urvp\CnDModel"
    r"\test_env\bpe_simple_vocab_16e6.txt.gz"
)

# Synonyms per class; candidates from all synonyms are pooled and de-duplicated.
PROMPTS = {
    "roof": ["building rooftop", "roof"],
    "facade": ["building facade", "building wall"],
}
# How many instances to merge into the "primary" mask per class.
TOP_K = {"roof": 1, "facade": 2}

COLORS_BGR = {"roof": (0, 0, 255), "facade": (255, 160, 0)}


@dataclass
class Cand:
    mask: np.ndarray          # bool, crop resolution
    score: float
    prompt: str
    contain: float = 0.0      # fraction of the mask inside the (dilated) island
    cover: float = 0.0        # fraction of the island explained by the mask
    rank: float = 0.0
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def find_islands(binary, min_area):
    """Connected components -> list of (id, (x1, y1, x2, y2)) with x2/y2 exclusive."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    islands = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            islands.append((i, (int(x), int(y), int(x + w), int(y + h))))
    return labels, islands


def square_crop(box, W, H, pad_frac, min_side):
    """Padded square crop around a box, shifted (not shrunk) to stay in-image."""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad_frac)
    side = int(round(min(max(side, min_side), W, H)))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x0 = min(max(x0, 0), W - side)
    y0 = min(max(y0, 0), H - side)
    return x0, y0, x0 + side, y0 + side


def iou(a, b):
    return (a & b).sum() / ((a | b).sum() + 1e-9)


def dilate(mask, radius):
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


# --------------------------------------------------------------------------
# SAM 3 wrappers
# --------------------------------------------------------------------------
def run_prompt(processor, state, prompt, out_hw):
    """Text prompt on an already-encoded image -> list[Cand] at out_hw."""
    if hasattr(processor, "reset_all_prompts"):
        processor.reset_all_prompts(state)
    out = processor.set_text_prompt(state=state, prompt=prompt)

    masks, scores = out.get("masks"), out.get("scores")
    if masks is None or masks.shape[0] == 0:
        return []
    masks = masks.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()

    cands = []
    for m, s in zip(masks, scores):
        m = np.squeeze(m).astype(np.float32)
        if m.shape != out_hw:  # we upscaled the crop before SAM; bring it back
            m = cv2.resize(m, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)
        cands.append(Cand(mask=m > 0.5, score=float(s), prompt=prompt))
    return cands


def dedupe(cands, iou_thr=0.85):
    kept = []
    for c in sorted(cands, key=lambda c: -c.score):
        if all(iou(c.mask, k.mask) < iou_thr for k in kept):
            kept.append(c)
    return kept


def select(cands, island, island_dil, min_contain, min_area):
    """Rank candidates by how well they match THIS island, not just by score."""
    island_area = max(island.sum(), 1)
    out = []
    for c in cands:
        a = c.mask.sum()
        if a < min_area:
            continue
        c.contain = (c.mask & island_dil).sum() / a
        c.cover = (c.mask & island).sum() / island_area
        if c.contain < min_contain:  # bleeds into neighbours / whole block
            continue
        c.rank = c.score * c.contain * np.sqrt(c.cover)
        if c.rank > 0:
            out.append(c)
    return sorted(out, key=lambda c: -c.rank)


# --------------------------------------------------------------------------
# One building
# --------------------------------------------------------------------------
def process_building(processor, image, island_full, bbox, args):
    W, H = image.size
    cx0, cy0, cx1, cy1 = square_crop(bbox, W, H, args.pad, args.min_side)
    crop = image.crop((cx0, cy0, cx1, cy1))
    ch, cw = crop.size[1], crop.size[0]

    island = island_full[cy0:cy1, cx0:cx1].astype(bool)
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    island_dil = dilate(island, max(3, int(0.15 * max(bw, bh))))
    min_area = max(20, int(0.03 * island.sum()))

    model_in = crop.resize((args.upscale, args.upscale), Image.BICUBIC) \
        if args.upscale else crop

    with torch.inference_mode():
        state = processor.set_image(model_in)  # encode once, reuse per prompt
        ranked = {}
        for cls, prompts in PROMPTS.items():
            cands = []
            for p in prompts:
                cands += run_prompt(processor, state, p, (ch, cw))
            ranked[cls] = select(
                dedupe(cands), island, island_dil, args.min_contain, min_area
            )

    # Roof wins: remove roof pixels from facade candidates, drop what's left over.
    roof_sel = ranked["roof"][: TOP_K["roof"]]
    roof_union = np.zeros((ch, cw), bool)
    for c in roof_sel:
        roof_union |= c.mask

    facade_ok = []
    for c in ranked["facade"]:
        before = c.mask.sum()
        c.mask = c.mask & ~roof_union
        if c.mask.sum() >= max(min_area, 0.4 * before):
            facade_ok.append(c)
    ranked["facade"] = facade_ok
    facade_sel = facade_ok[: TOP_K["facade"]]

    selected = {"roof": roof_sel, "facade": facade_sel}
    return (cx0, cy0, cx1, cy1), crop, island, ranked, selected


def union(cands, shape):
    m = np.zeros(shape, bool)
    for c in cands:
        m |= c.mask
    return m


def save_overlay(path, crop, island, masks):
    img = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    out = img.copy()
    for cls, m in masks.items():
        out[m] = COLORS_BGR[cls]
    out = cv2.addWeighted(img, 0.55, out, 0.45, 0)
    cnts, _ = cv2.findContours(
        island.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, cnts, -1, (0, 255, 255), 1)  # yellow = UNet island
    cv2.imwrite(path, out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--mask", help="UNet mask PNG (damaged/collapsed pixels)")
    ap.add_argument("--classes", type=int, nargs="*",
                    help="mask values to keep (default: any non-zero)")
    ap.add_argument("--box", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="single manual box instead of --mask (for debugging)")
    ap.add_argument("--checkpoint", default="sam3.pt")
    ap.add_argument("--bpe", default=DEFAULT_BPE)
    ap.add_argument("--out", default="inspection_results")
    ap.add_argument("--pad", type=float, default=0.5,
                    help="padding per side as a fraction of the island size")
    ap.add_argument("--min-side", type=int, default=96)
    ap.add_argument("--upscale", type=int, default=1008,
                    help="resize each crop to this square size before SAM (0=off)")
    ap.add_argument("--conf", type=float, default=0.2,
                    help="SAM 3 confidence threshold (selection filters further)")
    ap.add_argument("--min-island", type=int, default=40)
    ap.add_argument("--min-contain", type=float, default=0.6)
    args = ap.parse_args()

    if not args.mask and not args.box:
        ap.error("provide --mask or --box")
    for p in (args.image, args.checkpoint):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    image = Image.open(args.image).convert("RGB")
    W, H = image.size

    # ---- islands -----------------------------------------------------------
    if args.mask:
        m = np.array(Image.open(args.mask))
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        binary = np.isin(m, args.classes) if args.classes else m > 0
        labels, islands = find_islands(binary, args.min_island)
    else:
        x1, y1, x2, y2 = args.box
        if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
            raise ValueError(f"Bad box {args.box} for image {W}x{H}")
        labels = np.zeros((H, W), np.int32)
        labels[y1:y2, x1:x2] = 1
        islands = [(1, (x1, y1, x2, y2))]
    print(f"{len(islands)} building island(s)")

    os.makedirs(args.out, exist_ok=True)

    # ---- model -------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM 3 on {device}...")
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, bpe_path=args.bpe)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=args.conf)

    roof_labels = np.zeros((H, W), np.uint16)
    facade_labels = np.zeros((H, W), np.uint16)
    records = []

    for n, (lab_id, bbox) in enumerate(islands, start=1):
        bdir = os.path.join(args.out, f"b{n:03d}")
        os.makedirs(bdir, exist_ok=True)

        island_full = labels == lab_id
        (x0, y0, x1c, y1c), crop, island, ranked, selected = process_building(
            processor, image, island_full, bbox, args
        )
        shape = island.shape
        prim = {cls: union(sel, shape) for cls, sel in selected.items()}

        crop.save(os.path.join(bdir, "crop.png"))
        for cls, m in prim.items():
            cv2.imwrite(os.path.join(bdir, f"{cls}.png"), m.astype(np.uint8) * 255)
        save_overlay(os.path.join(bdir, "overlay.png"), crop, island, prim)

        roof_labels[y0:y1c, x0:x1c][prim["roof"]] = n
        facade_labels[y0:y1c, x0:x1c][prim["facade"]] = n

        rec = {
            "building_id": n,
            "island_bbox": list(bbox),
            "crop_box": [x0, y0, x1c, y1c],
            "instances": [],
        }
        for cls, cands in ranked.items():
            for k, c in enumerate(cands):
                rec["instances"].append({
                    "class": cls, "index": k, "primary": k < TOP_K[cls],
                    "score": round(c.score, 4), "prompt": c.prompt,
                    "contain": round(float(c.contain), 3),
                    "cover": round(float(c.cover), 3),
                    "rank": round(float(c.rank), 4),
                    "area_px": int(c.mask.sum()),
                })
        records.append(rec)
        print(f"[{n}/{len(islands)}] bbox={bbox} "
              f"roof={len(ranked['roof'])} facade={len(ranked['facade'])} cands")

    cv2.imwrite(os.path.join(args.out, "roof_labels.png"), roof_labels)
    cv2.imwrite(os.path.join(args.out, "facade_labels.png"), facade_labels)
    with open(os.path.join(args.out, "buildings.json"), "w") as f:
        json.dump(records, f, indent=2)
    print(f"Done. Results in {args.out}/")


if __name__ == "__main__":
    main()