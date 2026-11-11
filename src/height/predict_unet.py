#!/usr/bin/env python3
"""
Run the trained U-Net on the evaluation tiles and write label PNGs (0/1/2).

  py predict_unet.py --ckpt runs/unet_r34/best.pt                 # test split, 40 tiles
  py predict_unet.py --ckpt runs/unet_r34/best.pt --split val --n-tiles 0   # all val tiles

Use the SAME --split / --n-tiles / --seed in predict_sam3.py so both models are
scored on identical tiles.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image

from bandon_common import (ROOT, list_tiles, load_rgb, normalize, positions,
                           pred_name)


@torch.no_grad()
def predict_tile(model, img, win, stride, batch, device, use_amp, tta):
    H, W, _ = img.shape
    coords = [(y, x) for y in positions(H, win, stride) for x in positions(W, win, stride)]
    probs = np.zeros((3, H, W), np.float32)
    count = np.zeros((H, W), np.float32)
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        arr = np.stack([normalize(img[cy:cy + win, cx:cx + win]) for cy, cx in chunk])
        xb = torch.from_numpy(arr).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            p = torch.softmax(model(xb).float(), 1)
            if tta:
                p = (p + torch.softmax(model(xb.flip(-1)).float(), 1).flip(-1)) / 2
        p = p.cpu().numpy()
        for (cy, cx), pi in zip(chunk, p):
            h, w = pi.shape[1:]
            probs[:, cy:cy + h, cx:cx + w] += pi
            count[cy:cy + h, cx:cx + w] += 1
    return (probs / np.maximum(count, 1)).argmax(0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n-tiles", type=int, default=40, help="0 = all tiles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="preds/unet")
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tta", action="store_true", help="horizontal-flip test-time augmentation")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = smp.Unet(ck["args"]["encoder"], encoder_weights=None, in_channels=3, classes=3)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    tiles = list_tiles(args.root, args.split, args.n_tiles, args.seed)
    if not tiles:
        raise SystemExit("No tiles found.")
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(tiles)} tiles -> {out}")

    t0 = time.time()
    for i, d in enumerate(tiles, 1):
        pred = predict_tile(model, load_rgb(d["img"]), args.win, args.stride,
                            args.batch, device, use_amp, args.tta)
        Image.fromarray(pred).save(out / pred_name(d))
        if i % 5 == 0 or i == len(tiles):
            rate = (time.time() - t0) / i
            print(f"  {i}/{len(tiles)}  {rate:.1f}s/tile  eta {rate*(len(tiles)-i)/60:.1f} min", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
