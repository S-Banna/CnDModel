import os
import json
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from PIL import Image
import segmentation_models_pytorch as smp

from config import RUN_LIST

import sys
sys.path.append("../rubble")
from quantify import quantify_building # type: ignore


# =============================================================================
# IMAGE LOADING
# =============================================================================

def load_image(path):
    return np.array(
        Image.open(path).convert("RGB")
    ).astype(np.float32) / 255.0


def load_mask(path):
    if path is None:
        return None

    if not os.path.exists(path):
        return None

    return np.array(
        Image.open(path).convert("L")
    )


# =============================================================================
# GSD LOADING
# =============================================================================

def load_gsd(label_path=None, default=0.5):
    """
    Priority:
        label.json metadata.pan_resolution
        label.json metadata.gsd / 4
        default
    """

    if label_path is None:
        print(
            f"GSD not specified and no label provided.\n"
            f"Using default GSD = {default} m/px"
        )
        return default

    if not os.path.exists(label_path):
        print(
            f"Label file not found: {label_path}\n"
            f"Using default GSD = {default} m/px"
        )
        return default

    try:
        with open(label_path, "r") as f:
            meta = json.load(f)["metadata"]

        if "pan_resolution" in meta:
            return float(meta["pan_resolution"])

        if "gsd" in meta:
            return float(meta["gsd"]) / 4.0

    except Exception:
        pass

    print(
        f"Could not determine GSD from label.\n"
        f"Using default GSD = {default} m/px"
    )

    return default


# =============================================================================
# MODEL
# =============================================================================

def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=6,
        classes=1,
    ).to(device)

    model.load_state_dict(
        torch.load(
            "../segmentation/model.pth",
            map_location=device
        )
    )

    model.eval()

    return model, device


# =============================================================================
# PREDICTION
# =============================================================================

def predict(model, device, pre, post):
    stacked = np.concatenate(
        [pre, post],
        axis=2
    )

    tensor = (
        torch.tensor(stacked)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.no_grad():
        logits = model(tensor)

    probs = (
        torch.sigmoid(logits)
        .cpu()
        .squeeze()
        .numpy()
    )

    return probs


# =============================================================================
# ACCURACY METRICS  (mirrors train.py's final test-metric computation)
# =============================================================================

def compute_accuracy_metrics(binary, truth, mask_binarize_thresh=0):
    """
    binary: predicted binary mask, values in {0,1}, shape (H,W)
    truth:  ground-truth grayscale mask, values in [0,255], shape (H,W)

    Returns a dict with the same fields train.py reports at test time:
    TP, FP, TN, FN, precision, recall, f1, global_iou, oa, per_image_iou
    """

    pred = (binary > 0).astype(np.uint8)
    gt = (truth > mask_binarize_thresh).astype(np.uint8)

    if pred.shape != gt.shape:
        # Resize truth to match prediction if needed (e.g. different source resolution)
        gt = cv2.resize(
            gt,
            (pred.shape[1], pred.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    TP = int(np.sum((pred == 1) & (gt == 1)))
    FP = int(np.sum((pred == 1) & (gt == 0)))
    TN = int(np.sum((pred == 0) & (gt == 0)))
    FN = int(np.sum((pred == 0) & (gt == 1)))

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    global_iou = TP / (TP + FP + FN + 1e-6)
    oa = (TP + TN) / (TP + TN + FP + FN + 1e-6)

    # per-image IoU, computed the same way compute_iou() does in train.py
    intersection = TP
    union = TP + FP + FN
    per_image_iou = (intersection + 1e-6) / (union + 1e-6)

    return {
        "tp": TP,
        "fp": FP,
        "tn": TN,
        "fn": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "global_iou": global_iou,
        "per_image_iou": per_image_iou,
        "oa": oa,
    }


def write_accuracy_file(output_dir, metrics):
    path = os.path.join(output_dir, "acc.txt")

    with open(path, "w") as f:

        if metrics is None:
            f.write("ACCURACY METRICS\n")
            f.write("================\n\n")
            f.write("No ground-truth mask was provided for this sample.\n")
            f.write("Accuracy metrics could not be computed.\n")
            return

        f.write("ACCURACY METRICS\n")
        f.write("================\n\n")

        f.write(f"Per-image IoU: {metrics['per_image_iou']:.4f}\n")
        f.write(f"Global IoU:    {metrics['global_iou']:.4f}\n")
        f.write(f"Precision:     {metrics['precision']:.4f}\n")
        f.write(f"Recall:        {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:      {metrics['f1']:.4f}\n")
        f.write(f"Overall Accuracy (OA): {metrics['oa']:.4f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"TP: {metrics['tp']:,}\n")
        f.write(f"FP: {metrics['fp']:,}\n")
        f.write(f"TN: {metrics['tn']:,}\n")
        f.write(f"FN: {metrics['fn']:,}\n")


# =============================================================================
# OVERLAYS
# =============================================================================

def damage_overlay(post, binary):
    out = post.copy()
    out[binary == 1] = [1.0, 0.2, 0.2]
    return out


def border_overlay(post, binary):
    mask = (binary * 255).astype(np.uint8)

    border = mask - cv2.erode(
        mask,
        np.ones((3, 3), np.uint8),
        iterations=3
    )

    out = post.copy()
    out[border > 0] = [1.0, 0.2, 0.2]

    return out


# =============================================================================
# BUILDING IDS
# =============================================================================

def draw_ids(ax, buildings):
    for b in buildings:
        x, y = b["centroid"]

        txt = ax.text(
            x,
            y,
            str(b["id"]),
            color="yellow",
            fontsize=7,
            ha="center",
            va="center",
            fontweight="bold"
        )

        txt.set_path_effects([
            pe.Stroke(
                linewidth=2,
                foreground="black"
            ),
            pe.Normal()
        ])


# =============================================================================
# RUBBLE EXTRACTION
# =============================================================================

MIN_PIXELS = 40
CLOSE_METRES = 2.0


def extract_buildings(binary, gsd):
    binary255 = (binary * 255).astype(np.uint8)

    k = max(
        1,
        round(CLOSE_METRES / gsd)
    )

    kernel = np.ones(
        (k, k),
        np.uint8
    )

    cleaned = cv2.morphologyEx(
        binary255,
        cv2.MORPH_CLOSE,
        kernel
    )

    (
        num_labels,
        labels,
        stats,
        centroids
    ) = cv2.connectedComponentsWithStats(
        cleaned
    )

    buildings = []

    for i in range(1, num_labels):

        area = int(
            stats[i, cv2.CC_STAT_AREA]
        )

        if area < MIN_PIXELS:
            continue

        buildings.append({
            "id": i,
            "pixel_area": area,
            "centroid": (
                int(centroids[i][0]),
                int(centroids[i][1])
            )
        })

    return cleaned, buildings


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def get_output_folder(post_path):
    stem = os.path.splitext(
        os.path.basename(post_path)
    )[0]

    folder = "_".join(
        stem.split("_")[:2]
    )

    return folder


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(
    model,
    device,
    pre_path,
    post_path,
    mask_path=None,
    label_path=None,
    threshold=0.5,
    gsd=None,
    structure_type=None
):
    print(f"\nProcessing: {post_path}")

    if structure_type is None:
        structure_type = "Residential Low Rise"

        print(
            "Structure type not selected.\n"
            "Using default structure type = Residential Low Rise"
        )

    if gsd is None:
        gsd = load_gsd(label_path)

    print(
        f"GSD = {gsd:.3f} m/px"
    )

    pre = load_image(pre_path)
    post = load_image(post_path)
    truth = load_mask(mask_path)

    probs = predict(
        model,
        device,
        pre,
        post
    )

    binary = (
        probs > threshold
    ).astype(np.uint8)

    overlay = damage_overlay(
        post,
        binary
    )

    border = border_overlay(
        post,
        binary
    )

    # =========================================================================
    # ACCURACY METRICS
    # =========================================================================

    if truth is not None:
        acc_metrics = compute_accuracy_metrics(binary, truth)

        print(
            f"Accuracy vs mask -> "
            f"IoU: {acc_metrics['per_image_iou']:.4f}, "
            f"Precision: {acc_metrics['precision']:.4f}, "
            f"Recall: {acc_metrics['recall']:.4f}, "
            f"F1: {acc_metrics['f1']:.4f}, "
            f"OA: {acc_metrics['oa']:.4f}"
        )
    else:
        acc_metrics = None
        print("No ground-truth mask provided; skipping accuracy computation.")

    # =========================================================================
    # SEGMENTATION FIGURE
    # =========================================================================

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10)
    )

    axes[0, 0].imshow(pre)
    axes[0, 0].set_title("Pre")

    axes[0, 1].imshow(post)
    axes[0, 1].set_title("Post")

    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title("Damage Overlay")

    if truth is not None:

        axes[1, 0].imshow(
            truth,
            cmap="gray"
        )

    else:

        axes[1, 0].imshow(
            np.zeros(
                binary.shape,
                dtype=np.uint8
            ),
            cmap="gray"
        )

        h, w = binary.shape

        axes[1, 0].plot(
            [0, w],
            [0, h],
            "r",
            linewidth=4
        )

        axes[1, 0].plot(
            [0, w],
            [h, 0],
            "r",
            linewidth=4
        )

    axes[1, 0].set_title(
        "Ground Truth"
    )

    axes[1, 1].imshow(
        binary,
        cmap="gray"
    )

    axes[1, 1].set_title(
        "Prediction"
    )

    axes[1, 2].imshow(border)

    axes[1, 2].set_title(
        "Border Overlay"
    )

    for ax in axes.flatten():
        ax.axis("off")

    # =========================================================================
    # RUBBLE
    # =========================================================================

    cleaned, buildings = extract_buildings(
        binary,
        gsd
    )

    results = [
        quantify_building(
            building,
            gsd,
            structure_type
        )
        for building in buildings
    ]

    # =========================================================================
    # OUTPUT DIR
    # =========================================================================

    folder_name = get_output_folder(
        post_path
    )

    output_dir = os.path.join(
        "outputs",
        folder_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # =========================================================================
    # SAVE SEGMENTATION FIGURE
    # =========================================================================

    pipeline_path = os.path.join(
        output_dir,
        f"{folder_name}_pipeline.png"
    )

    fig.savefig(
        pipeline_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =========================================================================
    # RUBBLE IMAGE
    # =========================================================================

    fig2, ax = plt.subplots(
        figsize=(10, 10)
    )

    ax.imshow(
        cleaned,
        cmap="gray"
    )

    draw_ids(
        ax,
        buildings
    )

    ax.axis("off")

    ax.set_title(
        f"Detected Buildings\n"
        f"GSD = {gsd:.3f} m/px\n"
        f"Structure = {structure_type}"
    )

    rubble_path = os.path.join(
        output_dir,
        f"{folder_name}_rubble.png"
    )

    fig2.savefig(
        rubble_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =========================================================================
    # MASS CSV
    # =========================================================================

    mass_df = pd.DataFrame([
        {
            "Building ID": r["building_id"],
            "Area m2": r["area_m2"],
            "Built-up m2": r["built_up_m2"],
            "Height m": r["height_m"],
            "Rubble m3": r["rubble_volume_m3"],
            "Concrete kg": r["concrete_kg"],
            "Steel kg": r["steel_kg"],
            "Masonry kg": r["masonry_kg"],
            "Wood kg": r["wood_kg"],
            "Other kg": r["other_kg"],
            "Total kg": r["mass_kg"],
        }
        for r in results
    ])

    mass_df.to_csv(
        os.path.join(
            output_dir,
            "rubble_mass.csv"
        ),
        index=False
    )

    # =========================================================================
    # CLEANUP CSV
    # =========================================================================

    cleanup_df = pd.DataFrame([
        {
            "Building ID": r["building_id"],
            "Manual hrs": r["manual_sort_hrs"],
            "Excavator hrs": r["excavator_hrs"],
            "Loader hrs": r["loader_hrs"],
            "Total hrs": r["total_cleanup_hrs"],
            "Workdays": r["total_cleanup_days"],
        }
        for r in results
    ])

    cleanup_df.to_csv(
        os.path.join(
            output_dir,
            "rubble_cleanup.csv"
        ),
        index=False
    )

    # =========================================================================
    # ACCURACY TXT
    # =========================================================================

    write_accuracy_file(
        output_dir,
        acc_metrics
    )

    # =========================================================================
    # SUMMARY TXT
    # =========================================================================

    total_rubble = sum(
        r["rubble_volume_m3"]
        for r in results
    )

    total_mass = sum(
        r["mass_kg"]
        for r in results
    )

    total_days = sum(
        r["total_cleanup_days"]
        for r in results
    )

    with open(
        os.path.join(
            output_dir,
            "rubble_summary.txt"
        ),
        "w"
    ) as f:

        f.write(
            "RUBBLE SUMMARY\n"
        )

        f.write(
            "==============\n\n"
        )

        f.write(
            f"GSD: {gsd:.3f} m/px\n"
        )

        f.write(
            f"Structure Type: {structure_type}\n"
        )

        f.write(
            f"Detected Buildings: {len(results)}\n\n"
        )

        f.write(
            f"Total Rubble Volume: {total_rubble:.2f} m3\n"
        )

        f.write(
            f"Total Rubble Mass: {total_mass/1000:.2f} tonnes\n"
        )

        f.write(
            f"Estimated Cleanup Duration: {total_days:.2f} workdays\n"
        )

    print(
        f"Saved {output_dir}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pre")
    parser.add_argument("--post")
    parser.add_argument("--mask")
    parser.add_argument("--label")

    parser.add_argument(
        "--threshold",
        type=float,
        default=None
    )

    parser.add_argument(
        "--gsd",
        type=float,
        default=None
    )

    parser.add_argument(
        "--structure",
        default=None
    )

    parser.add_argument(
        "--batch",
        action="store_true"
    )

    args = parser.parse_args()

    if args.threshold is None:

        threshold = 0.5

        print(
            "Threshold not selected.\n"
            "Using default threshold = 0.5"
        )

    else:
        threshold = args.threshold

    model, device = load_model()

    if args.batch:

        for sample in RUN_LIST:

            run_pipeline(
                model=model,
                device=device,
                pre_path=sample["pre"],
                post_path=sample["post"],
                mask_path=sample.get("mask"),
                label_path=sample.get("label"),
                threshold=threshold,
                gsd=sample.get("gsd"),
                structure_type=sample.get("structure_type")
            )

    else:

        run_pipeline(
            model=model,
            device=device,
            pre_path=args.pre,
            post_path=args.post,
            mask_path=args.mask,
            label_path=args.label,
            threshold=threshold,
            gsd=args.gsd,
            structure_type=args.structure
        )


if __name__ == "__main__":
    main()