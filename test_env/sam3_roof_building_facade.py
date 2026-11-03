"""
Per-building roof / building / facade segmentation with SAM 3.

facade := building - roof          (no direct "facade" text prompt)

Rationale: SAM 3 knows "building" and "roof" far better than "facade", and the
whole-building mask is usually more reliable than a facade or roof-edge mask
(the roof/facade seam is the ambiguous part). Facade is what is left over.

Pipeline per UNet island
  1. padded square crop around the island (upscaled before SAM),
  2. roof: text prompts, pick the instance matching THIS island,
  3. building: text prompt, pick the instance matching the island AND
     containing the chosen roof,
  4. nest (building |= roof), subtract, open away thin edge slivers,
  5. save roof / building / facade masks + a 4-panel overview.

Every building gets a `status` so failures can be tallied across scenes.
"""
import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass

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

PROMPTS = {
    "roof": ["building rooftop", "roof"],
    "building": ["building"],
}
TOP_K = {"roof": 1, "building": 1}

# BGR
RED, GREEN, BLUE, YELLOW = (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)


@dataclass
class Cand:
    mask: np.ndarray          # bool, crop resolution
    score: float
    prompt: str
    contain: float = 0.0      # fraction of the mask inside the (dilated) island
    cover: float = 0.0        # fraction of the island explained by the mask
    roof_in: float = 1.0      # (building only) fraction of the roof it contains
    rank: float = 0.0


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def find_islands(binary, min_area):
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


def ellipse(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def dilate(mask, radius):
    return cv2.dilate(mask.astype(np.uint8), ellipse(radius)).astype(bool)


def clean_facade(f, open_px, min_area):
    """Remove thin edge slivers (two independent masks never agree exactly on
    the roof edge) and tiny leftover specks."""
    f = f.astype(np.uint8)
    if open_px > 0:
        f = cv2.morphologyEx(f, cv2.MORPH_OPEN, ellipse(open_px))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(f, connectivity=8)
    out = np.zeros(f.shape, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out |= lab == i
    return out


# --------------------------------------------------------------------------
# SAM 3 wrappers
# --------------------------------------------------------------------------
def run_prompt(processor, state, prompt, out_hw):
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
        if m.shape != out_hw:
            m = cv2.resize(m, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)
        cands.append(Cand(mask=m > 0.5, score=float(s), prompt=prompt))
    return cands


def dedupe(cands, iou_thr=0.85):
    kept = []
    for c in sorted(cands, key=lambda c: -c.score):
        if all(iou(c.mask, k.mask) < iou_thr for k in kept):
            kept.append(c)
    return kept


def select(cands, island, island_dil, min_contain, min_area,
           roof=None, min_roof_in=0.7):
    """Rank candidates by fit to THIS island (and, for buildings, to the roof)."""
    island_area = max(island.sum(), 1)
    have_roof = roof is not None and roof.any()
    out = []
    for c in cands:
        a = c.mask.sum()
        if a < min_area:
            continue
        c.contain = (c.mask & island_dil).sum() / a
        c.cover = (c.mask & island).sum() / island_area
        if c.contain < min_contain:
            continue
        c.rank = c.score * c.contain * np.sqrt(c.cover)
        if have_roof:
            c.roof_in = (c.mask & roof).sum() / roof.sum()
            if c.roof_in < min_roof_in:   # a building must contain its roof
                continue
            c.rank *= c.roof_in
        if c.rank > 0:
            out.append(c)
    return sorted(out, key=lambda c: -c.rank)


def union(cands, shape):
    m = np.zeros(shape, bool)
    for c in cands:
        m |= c.mask
    return m


# --------------------------------------------------------------------------
# One building
# --------------------------------------------------------------------------
def process_building(processor, image, island_full, bbox, args):
    W, H = image.size
    cx0, cy0, cx1, cy1 = square_crop(bbox, W, H, args.pad, args.min_side)
    crop = image.crop((cx0, cy0, cx1, cy1))
    ch, cw = crop.size[1], crop.size[0]
    shape = (ch, cw)

    island = island_full[cy0:cy1, cx0:cx1].astype(bool)
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    island_dil = dilate(island, max(3, int(0.15 * max(bw, bh))))
    min_area = max(20, int(0.03 * island.sum()))

    model_in = crop.resize((args.upscale, args.upscale), Image.BICUBIC) \
        if args.upscale else crop

    def candidates(state, cls):
        cands = []
        for p in PROMPTS[cls]:
            cands += run_prompt(processor, state, p, shape)
        return dedupe(cands)

    with torch.inference_mode():
        state = processor.set_image(model_in)  # encode once per crop

        roof_all = select(candidates(state, "roof"), island, island_dil,
                          args.min_contain, min_area)
        roof = union(roof_all[: TOP_K["roof"]], shape)

        bld_all = select(candidates(state, "building"), island, island_dil,
                         args.min_contain, min_area,
                         roof=roof, min_roof_in=args.min_roof_in)
        building = union(bld_all[: TOP_K["building"]], shape)

    # ---- status + minus step ----------------------------------------------
    have_roof, have_bld = roof.any(), building.any()
    if have_roof:
        building = building | roof        # enforce roof inside building
    if not have_roof and not have_bld:
        status = "no_roof_no_building"
    elif not have_roof:
        status = "no_roof"
    elif not have_bld:
        status = "no_building"            # building == roof, facade empty
    else:
        status = "ok"

    facade = np.zeros(shape, bool)
    if have_roof and have_bld:
        raw = building & ~roof
        facade = clean_facade(
            raw, args.open_px,
            max(15, int(args.min_facade_frac * building.sum())),
        )
        if not facade.any():
            status = "no_facade"

    return {
        "crop_box": (cx0, cy0, cx1, cy1), "crop": crop, "island": island,
        "roof": roof, "building": building, "facade": facade,
        "ranked": {"roof": roof_all, "building": bld_all}, "status": status,
    }


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def tint(img, mask, color, alpha=0.5):
    out = img.copy()
    out[mask] = color
    return cv2.addWeighted(img, 1 - alpha, out, alpha, 0)


def draw_contour(img, mask, color, thick=1):
    cnts, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img, cnts, -1, color, thick)


def save_panels(path, crop, island, roof, building, facade, size=400):
    """roof | building | building - roof | combined   (yellow = UNet island)"""
    base = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)

    p1 = tint(base, roof, RED)
    p2 = tint(base, building, GREEN)
    p3 = tint(base, facade, BLUE)
    p4 = tint(tint(base, facade, BLUE), roof, RED)
    draw_contour(p4, building, GREEN)

    panels = []
    titles = [f"roof {int(roof.sum())}px", f"building {int(building.sum())}px",
              f"building - roof {int(facade.sum())}px", "combined"]
    for p, t in zip((p1, p2, p3, p4), titles):
        draw_contour(p, island, YELLOW)
        p = cv2.resize(p, (size, size), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(p, (0, 0), (size, 22), (0, 0, 0), -1)
        cv2.putText(p, t, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(p)
    cv2.imwrite(path, np.hstack(panels))


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
    ap.add_argument("--pad", type=float, default=0.5)
    ap.add_argument("--min-side", type=int, default=96)
    ap.add_argument("--upscale", type=int, default=1008)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--min-island", type=int, default=100)
    ap.add_argument("--min-contain", type=float, default=0.6)
    ap.add_argument("--min-roof-in", type=float, default=0.7,
                    help="building must contain at least this fraction of the roof")
    ap.add_argument("--open-px", type=int, default=1,
                    help="opening radius to delete thin slivers in building-roof")
    ap.add_argument("--min-facade-frac", type=float, default=0.03,
                    help="drop facade pieces smaller than this fraction of the building")
    args = ap.parse_args()

    if not args.mask and not args.box:
        ap.error("provide --mask or --box")
    for p in (args.image, args.checkpoint):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    image = Image.open(args.image).convert("RGB")
    W, H = image.size

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM 3 on {device}...")
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, bpe_path=args.bpe)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=args.conf)

    full = {k: np.zeros((H, W), np.uint16) for k in ("roof", "building", "facade")}
    records, statuses = [], Counter()

    for n, (lab_id, bbox) in enumerate(islands, start=1):
        bdir = os.path.join(args.out, f"b{n:03d}")
        os.makedirs(bdir, exist_ok=True)

        r = process_building(processor, image, labels == lab_id, bbox, args)
        x0, y0, x1c, y1c = r["crop_box"]
        statuses[r["status"]] += 1

        r["crop"].save(os.path.join(bdir, "crop.png"))
        for k in ("roof", "building", "facade"):
            cv2.imwrite(os.path.join(bdir, f"{k}.png"), r[k].astype(np.uint8) * 255)
            full[k][y0:y1c, x0:x1c][r[k]] = n
        save_panels(os.path.join(bdir, "panels.png"), r["crop"], r["island"],
                    r["roof"], r["building"], r["facade"])

        bld_area = int(r["building"].sum())
        rec = {
            "building_id": n, "status": r["status"],
            "island_bbox": list(bbox), "crop_box": [x0, y0, x1c, y1c],
            "roof_px": int(r["roof"].sum()), "building_px": bld_area,
            "facade_px": int(r["facade"].sum()),
            "facade_frac": round(float(r["facade"].sum()) / bld_area, 3)
            if bld_area else 0.0,
            "instances": [],
        }
        for cls, cands in r["ranked"].items():
            for k, c in enumerate(cands):
                rec["instances"].append({
                    "class": cls, "index": k, "primary": k < TOP_K[cls],
                    "score": round(c.score, 4), "prompt": c.prompt,
                    "contain": round(float(c.contain), 3),
                    "cover": round(float(c.cover), 3),
                    "roof_in": round(float(c.roof_in), 3),
                    "rank": round(float(c.rank), 4),
                    "area_px": int(c.mask.sum()),
                })
        records.append(rec)
        print(f"[{n}/{len(islands)}] bbox={bbox} status={r['status']} "
              f"roof={rec['roof_px']} bldg={rec['building_px']} "
              f"facade={rec['facade_px']}")

    for k, arr in full.items():
        cv2.imwrite(os.path.join(args.out, f"{k}_labels.png"), arr)
    with open(os.path.join(args.out, "buildings.json"), "w") as f:
        json.dump(records, f, indent=2)
    print("Status summary:", dict(statuses))
    print(f"Done. Results in {args.out}/")


if __name__ == "__main__":
    main()
