# Construction & Demolition (C&D) Rubble Volume Quantification Model

Repository for the development of an automated system to estimate rubble volume and debris mass from satellite imagery following building collapse events.

---

## Project Overview

This project aims to build a computer vision pipeline that:

- Takes coordinates of affected buildings  
- Retrieves pre- and post-conflict satellite imagery  
- Detects and delineates structural damage  
- Estimates collapsed surface area  
- Estimates number of floors (≈ 3m per floor, via Street View or metadata)  
- Computes total debris volume and mass per building  
- *(Future work)* Estimates material composition using contextual data  

---

## Current Direction

The project has transitioned from early classical change-detection prototypes to a structured, learning-based approach using annotated satellite datasets.

Key components under development:

- Pre/Post image ingestion and normalization  
- Damage mask generation from polygon annotations  
- Segmentation model training (UNet-style architecture)  
- Pixel-level evaluation using overlap-based metrics (e.g., IoU, recall)  

Training data is stored locally due to size constraints and referenced via configuration files.

---

## Intended Pipeline

1. Input building coordinates  
2. Retrieve standardized satellite imagery  
3. Detect and delineate collapse regions  
4. Compute affected surface area  
5. Estimate building height  
6. Derive debris volume and mass  

---