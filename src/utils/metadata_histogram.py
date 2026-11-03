#!/usr/bin/env python3
"""
Scan label JSONs (xBD layout: <root>/<tier>/labels/*.json), pull the
"metadata" block, and report how much of the imagery is usable for
off-nadir height estimation.

Outputs (in --out):
  metadata.csv           one row per JSON (all scalar metadata fields)
  summary.txt            percentiles, % usable, per-disaster / per-sensor tables
  metadata_histograms.png  tile-level and scene-level histograms

Reusable for the Lebanon data: point --root at it and, if the field names
differ, use --angle-field / --pixel-field / --scene-field / --pattern.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_ROOT = r"C:\Users\user\Downloads\xbd-dataset\xbd"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_rows(root: Path, pattern: str):
    files = sorted(root.glob(pattern))
    print(f"Found {len(files)} JSON files under {root} ({pattern})")
    rows, skipped = [], 0
    for i, p in enumerate(files, 1):
        try:
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)["metadata"]
        except (OSError, ValueError, KeyError, TypeError):
            skipped += 1
            continue
        # keep scalar fields only (drops the big polygon lists)
        row = {k: v for k, v in meta.items() if not isinstance(v, (dict, list))}
        rel = p.relative_to(root)
        row["split"] = rel.parts[0] if len(rel.parts) > 1 else ""
        row["json_file"] = str(rel)
        rows.append(row)
        if i % 5000 == 0:
            print(f"  ...{i}/{len(files)}")
    return pd.DataFrame(rows), skipped, len(files)


def prepare(df, args):
    name = df.get("img_name", pd.Series(index=df.index, dtype=object))
    name = name.fillna(df["json_file"].map(lambda s: Path(s).stem)).astype(str)
    df["stage"] = np.select(
        [name.str.contains("_pre_"), name.str.contains("_post_")],
        ["pre", "post"], default="unknown",
    )
    df["angle"] = pd.to_numeric(df.get(args.angle_field), errors="coerce")
    px = pd.to_numeric(df.get(args.pixel_field), errors="coerce")
    if args.pixel_field != "gsd" and "gsd" in df:
        px = px.fillna(pd.to_numeric(df["gsd"], errors="coerce"))
    df["pixel_m"] = px
    for c in ("disaster", "sensor"):
        if c not in df:
            df[c] = "unknown"
    scene = df.get(args.scene_field)
    fallback = df["disaster"].astype(str) + "|" + df.get(
        "capture_date", pd.Series("", index=df.index)).astype(str)
    df["scene"] = scene.fillna(fallback) if scene is not None else fallback

    ok = df["angle"] > 0
    # 1-pixel error in roof-footprint offset -> height error (metres)
    df["err_m_per_px"] = np.where(ok, df["pixel_m"] / np.tan(np.radians(df["angle"].where(ok, 45))), np.nan)
    df["usable"] = (df["angle"] >= args.min_angle) & (df["pixel_m"] <= args.max_pixel)
    return df


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def pct(x):
    return f"{100 * x:5.1f}%"


def group_table(d, key, thr):
    g = d.groupby(key)
    t = pd.DataFrame({
        "tiles": g.size(),
        "scenes": g["scene"].nunique(),
        "median_angle": g["angle"].median().round(1),
        "%>=thr": (g["angle"].apply(lambda s: (s >= thr).mean()) * 100).round(1),
        "%usable": (g["usable"].mean() * 100).round(1),
    })
    return t.sort_values("tiles", ascending=False)


def build_summary(df, d, skipped, total, args):
    scenes = d.drop_duplicates("scene")
    q = [0, 5, 25, 50, 75, 95, 100]
    lines = [
        f"Files found: {total}   parsed: {len(df)}   skipped: {skipped}",
        f"Subset analysed: stage={args.which}  ->  {len(d)} tiles, {len(scenes)} scenes",
        f"Angle field: {args.angle_field}   pixel-size field: {args.pixel_field} (falls back to gsd)",
        f"Usable = angle >= {args.min_angle} deg AND pixel size <= {args.max_pixel} m",
        "",
        "Off-nadir angle percentiles (deg)   " + "  ".join(f"p{p}" for p in q),
        "  tiles : " + "  ".join(f"{v:5.1f}" for v in np.nanpercentile(d["angle"].dropna(), q)),
        "  scenes: " + "  ".join(f"{v:5.1f}" for v in np.nanpercentile(scenes["angle"].dropna(), q)),
        "",
        f"Tiles  with angle >= {args.min_angle}: {pct((d['angle'] >= args.min_angle).mean())}"
        f"   usable: {pct(d['usable'].mean())}",
        f"Scenes with angle >= {args.min_angle}: {pct((scenes['angle'] >= args.min_angle).mean())}"
        f"   usable: {pct(scenes['usable'].mean())}",
        "",
        "Pixel size (m): " + "  ".join(
            f"p{p}={v:.2f}" for p, v in zip([5, 50, 95], np.nanpercentile(d["pixel_m"].dropna(), [5, 50, 95]))),
        "Height error per 1 px of offset (m): " + "  ".join(
            f"p{p}={v:.2f}" for p, v in zip([5, 50, 95], np.nanpercentile(d["err_m_per_px"].dropna(), [5, 50, 95]))),
        "",
        "Counts by stage (all files): " + str(df["stage"].value_counts().to_dict()),
        "",
        "By disaster:",
        group_table(d, "disaster", args.min_angle).to_string(),
        "",
        "By sensor:",
        group_table(d, "sensor", args.min_angle).to_string(),
    ]
    return "\n".join(lines)


def make_plots(d, path, args):
    scenes = d.drop_duplicates("scene")
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    thr = args.min_angle

    def hist(a, s, bins, title, xlabel, vline=None, color="tab:blue"):
        s = s.dropna()
        a.hist(s, bins=bins, color=color, edgecolor="white")
        if vline is not None:
            a.axvline(vline, color="red", ls="--", lw=1.5)
        a.set_title(f"{title}  (n={len(s)})")
        a.set_xlabel(xlabel)
        a.set_ylabel("count")

    hist(ax[0, 0], d["angle"], np.arange(0, 61, 2.5),
         "Off-nadir angle, tiles", "degrees", thr)
    hist(ax[0, 1], d["pixel_m"], 40, "Pixel size, tiles", "metres",
         args.max_pixel, "tab:green")
    hist(ax[0, 2], d["err_m_per_px"].clip(upper=10), 40,
         "Height error per 1 px of offset (clipped at 10 m)", "metres", None, "tab:orange")
    hist(ax[1, 0], scenes["angle"], np.arange(0, 61, 2.5),
         "Off-nadir angle, scenes (unique catalog id)", "degrees", thr)
    hist(ax[1, 2], d["target_azimuth"] if "target_azimuth" in d else pd.Series(dtype=float),
         np.arange(0, 361, 15), "Target azimuth, tiles", "degrees", None, "tab:purple")

    t = group_table(d, "disaster", thr)["%usable"].sort_values()
    ax[1, 1].barh(t.index, t.values, color="tab:gray")
    ax[1, 1].set_title(f"% tiles usable by disaster (angle>={thr}, px<={args.max_pixel} m)")
    ax[1, 1].set_xlim(0, 100)
    ax[1, 1].tick_params(axis="y", labelsize=8)

    fig.suptitle(f"Imagery metadata audit  [stage={args.which}]", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--pattern", default="**/labels/*.json")
    ap.add_argument("--out", default="metadata_report")
    ap.add_argument("--which", choices=["pre", "post", "both"], default="pre",
                    help="which imagery stage to analyse (height comes from PRE)")
    ap.add_argument("--min-angle", type=float, default=25.0)
    ap.add_argument("--max-pixel", type=float, default=1.0,
                    help="max usable pixel size in metres")
    ap.add_argument("--angle-field", default="off_nadir_angle")
    ap.add_argument("--pixel-field", default="pan_resolution",
                    help="xBD tiles are pan-sharpened, so pan_resolution is "
                         "probably the real pixel size; use 'gsd' to compare")
    ap.add_argument("--scene-field", default="catalog_id")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root not found: {root}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, skipped, total = load_rows(root, args.pattern)
    if df.empty:
        raise SystemExit("No metadata parsed. Check --root / --pattern.")
    df = prepare(df, args)
    df.to_csv(out / "metadata.csv", index=False)

    d = df if args.which == "both" else df[df["stage"] == args.which]
    d = d.dropna(subset=["angle", "pixel_m"])
    if d.empty:
        raise SystemExit(f"No rows left for stage={args.which}.")

    text = build_summary(df, d, skipped, total, args)
    print("\n" + text)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    make_plots(d, out / "metadata_histograms.png", args)
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()