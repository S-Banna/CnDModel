import cv2
import numpy as np


def clean_prediction(binary_mask):
    mask = (binary_mask * 255).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def extract_buildings(mask, min_pixels=40):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    buildings = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_pixels:
            continue

        building_mask = (labels == i)

        buildings.append({
            "id": i,
            "pixel_area": int(area),
            "mask": building_mask
        })

    return buildings