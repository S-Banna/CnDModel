#!/usr/bin/env python3
"""
Score any number of prediction folders against BANDON ground truth and make
side-by-side comparison images.

  py score.py --preds unet=preds/unet sam3=preds/sam3
  py score.py --preds unet=preds/unet sam3=preds/sam3 --split val --n-tiles 40

Only tiles that EVERY model has a prediction for are scored (fair comparison).
Metrics are dataset-level (pixels pooled over all tiles):
  IoU roof, IoU facade, IoU building (roof U facade), mean IoU of roof+facade.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bandon_common import (FACADE_RGB, ROOF_RGB, ROOT, ConfMat, list_tiles,
                           load_labels, load_rgb, pred_name)


def overlay(img, lab, alpha=0.55):
    out = img.copy()
    out[lab == 1] = ROOF_RGB
    out[lab == 2] = FACADE_RGB
    return cv2.addWeighted(img, 1 - alpha, out, alpha, 0)


def titled(rgb, text, size):
    p = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(p, (0, 0), (size, 24), (0, 0, 0), -1)
    cv2.putText(p, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n-tiles", type=int, default=40, help="must match the predict scripts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preds", nargs="+", required=True, metavar="NAME=DIR",
                    help="prediction roots (the split name is appended automatically)")
    ap.add_argument("--panels", type=int, default=6, help="comparison images to save")
    ap.add_argument("--out", default="compare")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    models = {}
    for item in args.preds:
        name, _, folder = item.partition("=")
        models[name] = Path(folder) / args.split

    tiles = list_tiles(args.root, args.split, args.n_tiles, args.seed)
    common = [d for d in tiles if all((m / pred_name(d)).exists() for m in models.values())]
    print(f"{len(tiles)} tiles in list, {len(common)} have predictions from every model")
    if not common:
        raise SystemExit("Nothing to score. Check --split / --n-tiles / --preds paths.")

    cms = {n: ConfMat(3) for n in models}
    facade_frac = []
    for d in common:
        gt = load_labels(d["lab"])
        facade_frac.append(float((gt == 2).mean()))
        for n, folder in models.items():
            cms[n].add(gt, np.array(Image.open(folder / pred_name(d))))

    results = {n: cm.scores() for n, cm in cms.items()}
    cols = [("iou_roof", "IoU roof"), ("iou_facade", "IoU facade"),
            ("iou_building", "IoU building"), ("miou_roof_facade", "mIoU(r,f)"),
            ("f1_roof", "F1 roof"), ("f1_facade", "F1 facade")]
    print(f"\nsplit={args.split}  tiles={len(common)}")
    print(f"{'model':<10}" + "".join(f"{label:>13}" for _, label in cols))
    for n, r in results.items():
        print(f"{n:<10}" + "".join(f"{r[k]:>13.3f}" for k, _ in cols))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"scores_{args.split}.json").write_text(json.dumps(
        {"split": args.split, "tiles": len(common), "results": results}, indent=2))

    # panels: tiles with the most facade (most informative for this project)
    order = np.argsort(facade_frac)[::-1][:args.panels]
    for idx in order:
        d = common[int(idx)]
        img, gt = load_rgb(d["img"]), load_labels(d["lab"])
        panels = [titled(img, "image", args.size),
                  titled(overlay(img, gt), "ground truth", args.size)]
        for n, folder in models.items():
            pred = np.array(Image.open(folder / pred_name(d)))
            panels.append(titled(overlay(img, pred), n, args.size))
        strip = cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / f"{args.split}_{d['city']}_{d['t']}_{d['stem']}.png"), strip)
    print(f"\nsaved scores + {len(order)} comparison images to {out}/")
    print("legend: orange-red = roof, cyan = facade")


if __name__ == "__main__":
    main()
