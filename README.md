Repository for the Construction and Demolition (C&D) Rubble Volume Quantification Automated Model.

# Satellite Image Collapse Detection

In this project, we prepared a normalized Google Earth dataset.
For each location, we manually extracted pre-strike and post-strike satellite images with identical parameters:

- Fixed coordinate
- Same zoom level
- Vertical top-down view / side view
- Same field of view & crop
- Clean map style (no labels/UI)

This made the only difference between images the actual physical change, allowing the SSIM-based change detector to produce accurate collapse masks.

We also implemented:

1. `collapsedetection.py`:
   - Aligns images and removes blue annotations.
   - Detects changes using SSIM.
2. `driver.py`:
   - Loads images and saves the detected mask.
3. `patchcreation.py`:
   - Splits large images into smaller patches for easier processing.
