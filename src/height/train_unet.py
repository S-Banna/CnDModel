#!/usr/bin/env python3
"""
Train a ResNet34 U-Net on BANDON roof / facade labels (3 classes).

  pip install segmentation-models-pytorch
  py train_unet.py                      # defaults
  py train_unet.py --resume runs/unet_r34/last.pt

Design notes
  * trains on random 512x512 crops of the 2048x2048 tiles
  * batch 4 + gradient accumulation (x4 = effective 16) because batch norm is
    noisy at batch 4 and you only have 6 GB
  * holds out ~10% of LOCATIONS (all dates together) for model selection;
    the official val/test splits are left untouched for the final comparison
  * best.pt is saved whenever the held-out mean IoU(roof, facade) improves,
    so you can stop the run at any time
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bandon_common import (ROOT, ConfMat, list_tiles, load_labels, load_rgb,
                           normalize, split_train_holdout)


class CropDataset(Dataset):
    def __init__(self, tiles, crop, length, train, seed=0):
        self.tiles, self.crop, self.length = tiles, crop, length
        self.train, self.seed = train, seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        d = self.tiles[i % len(self.tiles)]
        rng = np.random.default_rng() if self.train else np.random.default_rng(self.seed + i)
        img, lab = load_rgb(d["img"]), load_labels(d["lab"])
        H, W = lab.shape
        c = self.crop
        for _ in range(6):                       # prefer crops that contain buildings
            y = int(rng.integers(0, max(H - c, 0) + 1))
            x = int(rng.integers(0, max(W - c, 0) + 1))
            patch = lab[y:y + c, x:x + c]
            if (patch > 0).mean() > 0.02:
                break
        img, lab = img[y:y + c, x:x + c], lab[y:y + c, x:x + c]

        if self.train:
            if rng.random() < 0.5:
                img, lab = img[:, ::-1], lab[:, ::-1]
            if rng.random() < 0.5:
                img, lab = img[::-1], lab[::-1]
            k = int(rng.integers(0, 4))
            img, lab = np.rot90(img, k), np.rot90(lab, k)
            a, b = rng.uniform(0.85, 1.15), rng.uniform(-15, 15)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

        lab = np.where(lab > 2, 255, lab)        # anything unexpected -> ignore
        x = torch.from_numpy(normalize(np.ascontiguousarray(img)))
        y = torch.from_numpy(np.ascontiguousarray(lab).astype(np.int64))
        return x, y


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    cm = ConfMat(3)
    for x, y in loader:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            logits = model(x.to(device, non_blocking=True))
        cm.add(y.numpy(), logits.argmax(1).cpu().numpy())
    return cm.scores()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default="runs/unet_r34")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--crops-per-tile", type=int, default=2,
                    help="an 'epoch' = this many random crops per training tile")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--class-weights", type=float, nargs=3, default=[1.0, 2.0, 3.0],
                    help="CE weights for background roof facade (facade is rare)")
    ap.add_argument("--holdout-pct", type=int, default=10)
    ap.add_argument("--val-crops", type=int, default=240)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    tiles = list_tiles(args.root, "train")
    if not tiles:
        raise SystemExit(f"No training tiles found under {args.root}\\train")
    train_tiles, hold_tiles = split_train_holdout(tiles, args.holdout_pct)
    print(f"device={device}  train tiles={len(train_tiles)}  held-out tiles={len(hold_tiles)}")
    if not hold_tiles:
        raise SystemExit("Held-out set is empty; raise --holdout-pct")

    train_ds = CropDataset(train_tiles, args.crop, len(train_tiles) * args.crops_per_tile, True)
    val_ds = CropDataset(hold_tiles, args.crop, args.val_crops, False, seed=123)
    kw = dict(num_workers=args.workers, pin_memory=use_amp,
              persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False, **kw)

    model = smp.Unet(args.encoder,
                     encoder_weights=None if args.no_pretrained else "imagenet",
                     in_channels=3, classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(train_dl) / args.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.05)
    scaler = make_scaler(use_amp)

    ce = nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights, device=device),
                             ignore_index=255)
    dice = smp.losses.DiceLoss("multiclass", ignore_index=255)

    start, best = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {start}, best={best:.4f}")

    log_path = out / "log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,loss,iou_roof,iou_facade,iou_building,miou_roof_facade,minutes\n")

    t0 = time.time()
    for epoch in range(start, args.epochs):
        model.train()
        running, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            logits = logits.float()
            loss = ce(logits, y) + dice(logits, y)
            scaler.scale(loss / args.accum).backward()
            if (i + 1) % args.accum == 0 or (i + 1) == len(train_dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            running += loss.item()
            n += 1
            if (i + 1) % 100 == 0:
                print(f"  epoch {epoch+1} it {i+1}/{len(train_dl)} loss {running/n:.4f}", flush=True)

        row = {"iou_roof": "", "iou_facade": "", "iou_building": "", "miou_roof_facade": ""}
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            s = validate(model, val_dl, device, use_amp)
            row = {k: f"{s[k]:.4f}" for k in row}
            if s["miou_roof_facade"] > best:
                best = s["miou_roof_facade"]
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "epoch": epoch, "score": best}, out / "best.pt")
                print("  * new best")
        mins = (time.time() - t0) / 60
        print(f"epoch {epoch+1}/{args.epochs}  loss {running/max(n,1):.4f}  "
              f"roof {row['iou_roof']} facade {row['iou_facade']} "
              f"building {row['iou_building']}  [{mins:.1f} min]", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{running/max(n,1):.4f}", row["iou_roof"],
                                    row["iou_facade"], row["iou_building"],
                                    row["miou_roof_facade"], f"{mins:.1f}"])
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best, "args": vars(args)}, out / "last.pt")
    print(f"done. best held-out mIoU(roof,facade) = {best:.4f}  ->  {out/'best.pt'}")


if __name__ == "__main__":
    main()
