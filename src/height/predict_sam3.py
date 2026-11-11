#!/usr/bin/env python3
"""
Zero-shot SAM 3 ("as-is": pretrained weights, text prompts only) on the
evaluation tiles, written as label PNGs (0 background, 1 roof, 2 facade).

  py predict_sam3.py                       # test split, 40 tiles (same subset as predict_unet.py)
  py predict_sam3.py --n-tiles 10          # quick smoke test

How a 2048x2048 tile becomes a label map
  1. slide a 512x512 window (stride 448); each window is upscaled to 1008x1008
     because tiny buildings are hopeless at 2048->1008 downscaling,
  2. prompts "building rooftop" + "roof"  -> roof   = union of instance masks
     prompt  "building"                   -> building = union of instance masks
  3. building |= roof ; facade = building - roof (thin slivers removed)
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from bandon_common import ROOT, list_tiles, load_rgb, positions, pred_name
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_ENV_DIR = _PROJECT_ROOT / "test_env"

DEFAULT_BPE = str(_ENV_DIR / "bpe_simple_vocab_16e6.txt.gz")
DEFAULT_CHECKPOINT = str(_ENV_DIR / "sam3.pt")

ROOF_PROMPTS = ["building rooftop", "roof"]
BUILDING_PROMPTS = ["building"]


def union_of_prompts(processor, state, prompts, size, min_score):
    out_mask = np.zeros((size, size), bool)
    for p in prompts:
        if hasattr(processor, "reset_all_prompts"):
            processor.reset_all_prompts(state)
        out = processor.set_text_prompt(state=state, prompt=p)
        masks, scores = out.get("masks"), out.get("scores")
        if masks is None or masks.shape[0] == 0:
            continue
        masks = masks.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        for m, s in zip(masks, scores):
            if s < min_score:
                continue
            m = np.squeeze(m).astype(np.float32)
            if m.shape != (size, size):
                m = cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)
            out_mask |= m > 0.5
    return out_mask


def predict_tile(processor, img, win, stride, upscale, min_score, open_px):
    H, W, _ = img.shape
    roof = np.zeros((H, W), bool)
    building = np.zeros((H, W), bool)
    for y in positions(H, win, stride):
        for x in positions(W, win, stride):
            crop = Image.fromarray(img[y:y + win, x:x + win])
            if upscale:
                crop = crop.resize((upscale, upscale), Image.BICUBIC)
            state = processor.set_image(crop)           # encode once per window
            r = union_of_prompts(processor, state, ROOF_PROMPTS, win, min_score)
            b = union_of_prompts(processor, state, BUILDING_PROMPTS, win, min_score)
            roof[y:y + win, x:x + win] |= r
            building[y:y + win, x:x + win] |= b
    building |= roof
    facade = building & ~roof
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
        facade = cv2.morphologyEx(facade.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    label = np.zeros((H, W), np.uint8)
    label[facade] = 2
    label[roof] = 1
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n-tiles", type=int, default=40, help="0 = all tiles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="preds/sam3")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--bpe", default=DEFAULT_BPE)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--stride", type=int, default=448)
    ap.add_argument("--upscale", type=int, default=1008)
    ap.add_argument("--conf", type=float, default=0.3, help="SAM 3 confidence threshold")
    ap.add_argument("--open-px", type=int, default=1)
    args = ap.parse_args()

    tiles = list_tiles(args.root, args.split, args.n_tiles, args.seed)
    if not tiles:
        raise SystemExit("No tiles found.")
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM 3 on {device}...")
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, bpe_path=args.bpe)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=args.conf)
    print(f"{len(tiles)} tiles -> {out}")

    t0 = time.time()
    with torch.inference_mode():
        for i, d in enumerate(tiles, 1):
            label = predict_tile(processor, load_rgb(d["img"]), args.win, args.stride,
                                 args.upscale, args.conf, args.open_px)
            Image.fromarray(label).save(out / pred_name(d))
            rate = (time.time() - t0) / i
            print(f"  {i}/{len(tiles)}  {rate:.1f}s/tile  eta {rate*(len(tiles)-i)/60:.1f} min",
                  flush=True)
    print("done.")


if __name__ == "__main__":
    main()
