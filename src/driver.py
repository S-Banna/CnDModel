import os
import collapsedetection as cd
import cv2

# Paths
pre_path = "../data/images_coords/images/single/Single08PreHaretHreik.png"
post_path = "../data/images_coords/images/single/Single08HaretHreik.png"

# Make sure results folder exists
os.makedirs("results", exist_ok=True)

mask = cd.change_detection(pre_path, post_path)

if mask is not None:
    cv2.imwrite("results/damage_mask01.png", mask)
    print("Mask saved successfully to results/damage_mask01.png")
else:
    print("No mask generated — check image paths or processing steps.")