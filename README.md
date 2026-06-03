# Collapse & Damage Detection Model
## Technical Progress Report

**May 2026**

---

**Project:** Satellite-based building damage segmentation and rubble volume quantification for conflict and disaster zones

**Repository:** github.com/S-Banna/CnDModel

**Status:** Active Development — Model training ongoing

**Report Period:** October 2025 – May 2026

---

# 1. Project Overview

This project develops an automated satellite-imagery analysis pipeline for rapid assessment of structural damage and construction-and-demolition (C&D) debris following disasters and armed conflicts. The broader objective is to support scalable post-conflict recovery and humanitarian response efforts by reducing reliance on slow, manual, and resource-intensive damage assessment methodologies. Conventional field-based assessment workflows are often difficult to conduct rapidly or safely in heavily damaged or conflict-affected environments, particularly at regional scale.

The current phase focuses on deep learning-based pixel-level segmentation of collapsed or heavily damaged structures from paired pre-event and post-event satellite imagery. These segmentation outputs form the foundation for a planned second-stage debris quantification pipeline intended to estimate rubble volume, material composition, and reconstruction burden at building and neighbourhood scales.

# 2. Phase 1: Exploration & Prototyping (Oct – Dec 2025)

The project was initiated in October 2025 with an exploratory phase focused on understanding the problem space and establishing a working data pipeline. Early work consisted of classical image processing approaches, repository organisation, and initial data collection efforts.

## 2.1 Classical Change Detection Experiments

Initial experiments employed simple pixel-differencing between pre/post image pairs as a baseline change detection approach. Prototype scripts were written to compare image pairs and visualise change heatmaps. While these methods confirmed that meaningful signal exists between pre/post pairs, they demonstrated fundamental limitations: sensitivity to lighting variation, seasonal change, and camera angle differences caused high false positive rates on non-damaged regions. These experiments were archived into a legacy/ directory and the project transitioned to a supervised deep learning approach.

## 2.2 Initial Data Collection

Early data collection encountered significant obstacles. Access to commercial satellite data providers (including Maxar and NASA portals) was either cost-prohibitive or operationally restricted. As an interim measure, 23 image pairs were manually collected from Google Earth, capturing pre/post views of conflict-affected urban areas in the target region. These images were manually annotated using the LabelMe annotation tool, with damaged building regions labelled as binary masks. This dataset, though limited in scale, served as the first domain-specific training resource and is currently incorporated directly into the training pool alongside xBD samples. A separate holdout domain dataset for cross-domain evaluation is pending transfer from a collaborating researcher.

Image pairs were preprocessed to align spatial extents, normalise resolution, and convert annotation masks from RGB colour coding (LabelMe default) to integer label format compatible with the training pipeline. Damage regions were encoded as pixel value 4 (matching the xBD damage label convention) with background encoded as 0.

## 2.3 Repository & Infrastructure Setup

Concurrent with data collection, the repository was restructured for collaborative development. A configuration system using config.yaml was introduced to decouple data paths from code, enabling the pipeline to run across different machines without code modification. Large data files and model weights were excluded from version control via .gitignore. The codebase was separated into src/ (model and training code) and data/ (configuration and local data references).

# 3. Phase 2: Architecture & Dataset Selection (Jan – Feb 2026)

## 3.1 Architecture Decision: U-Net

Following the classical approach being discarded, the project adopted a U-Net encoder-decoder convolutional architecture for semantic segmentation. The U-Net was selected on the basis of several properties well-suited to this task:

- Skip connections between encoder and decoder preserve spatial detail lost during downsampling, which is critical for accurate building boundary delineation at pixel level.
- The architecture is well-established for binary segmentation tasks on satellite and aerial imagery, with strong prior literature supporting its efficacy.
- It accepts arbitrary-channel inputs, making the 6-channel stacked pre/post tensor (2 × RGB) a natural fit without architectural modification.
- It is computationally tractable on consumer GPU hardware, enabling rapid iteration and practical experimentation under limited compute constraints.

The initial implementation comprised a three-level encoder (64, 128, 256 channels), a 512-channel bottleneck, and a symmetric decoder with transposed convolution upsampling and skip connections. All weights were randomly initialised and trained from scratch in the initial experiments.

## 3.2 Custom Dataset Implementation

A PyTorch Dataset class (XVDataset) was implemented to handle the pre/post image pair loading, mask parsing, normalisation, and random cropping. Key design decisions included:

- Input stacking: Pre and post images are concatenated channel-wise along the third axis to produce a single (H, W, 6) tensor, which is transposed to (6, H, W) for the model. This formulation allows the model to learn change features jointly across both temporal states.
- 256x256 random cropping: Training on random 256x256 crops extracted from 1024x1024 source images provides data augmentation, reduces GPU memory requirements, and substantially increases the number of effective training samples per source image.
- Damage-biased cropping: A configurable probability (default 0.5) controls the fraction of crops that are centred on a randomly selected damage pixel. This mitigates the class imbalance inherent in disaster imagery, where damaged pixels are a small spatial minority.
- Damage-only filtering: An optional flag (damage_only=True) filters the dataset to only include source images containing at least one damage pixel, eliminating uninformative all-background samples from the training pool.

# 4. Phase 3: Training Pipeline & Loss Formulation (Feb – Mar 2026)

## 4.1 Training Infrastructure

A full supervised training loop was implemented with the following components:

- Optimiser: Adam with initial learning rate 1e-4.
- Learning rate scheduling: ReduceLROnPlateau monitoring validation IoU, with a 0.5 decay factor and patience of 5 epochs. This adaptively halves the learning rate when validation performance stagnates, observed to trigger effectively at epochs 11 and 22 in early runs.
- Batch size: 8, with num_workers=4 and pin_memory=True for GPU data transfer efficiency.
- Model checkpointing: The model state is saved whenever a new peak validation IoU is observed, ensuring the best-performing checkpoint is retained regardless of final epoch performance.
- Validation split: Initially a random 10% hold-out from the training pool; later replaced by the dedicated hold subset from the xBD dataset for a more rigorous, independently-curated evaluation set.

## 4.2 Loss Function

Training initially used Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss) as the sole objective. This was subsequently augmented with a Dice Loss term:

**L_total = L_BCE + L_Dice**

The BCE term provides stable pixel-wise gradient signal and handles class imbalance via a configurable pos_weight parameter (set to 10.0), up-weighting the gradient contribution of damage pixels relative to background. The Dice loss term directly optimises the region overlap metric, penalising the model for predicting damage regions that are spatially offset from ground truth even when overall pixel accuracy is high. This combined objective is standard practice in medical and remote sensing segmentation literature for sparse positive class problems. In many training samples, damaged regions occupy only a small fraction of total image area, making class imbalance mitigation critical for stable optimisation.

## 4.3 Overfit Validation

Prior to full training, a systematic overfit test was conducted on two damage-containing image crops. The model was trained for 200 epochs at learning rate 1e-3 on fixed samples. The model reached a loss of 0.03 and IoU of 0.94 by epoch 200, confirming that the architecture, loss formulation, and data pipeline were all functioning correctly. This test was critical in ruling out silent bugs in the pipeline before committing to full training runs.

# 5. Phase 4: Dataset Scaling & Architecture Upgrade (Mar – May 2026)

## 5.1 xBD Dataset Integration

The xView2 Building Damage Assessment (xBD) dataset was identified as the primary training resource. xBD is a large-scale, publicly available dataset comprising paired pre/post satellite imagery across 19 disaster events worldwide, with pixel-level damage annotations derived from expert building polygon assessments. The dataset provides integer-valued grayscale masks where pixel values encode damage severity on a scale of 0-4, with values 3 and 4 corresponding to major and destroyed damage classes respectively, which are used as the positive class in the binary segmentation formulation.

The full xBD dataset (~11,000 image pairs across all disaster types) was obtained. The dataset is partitioned into tier1 (high-confidence labels), tier3 (lower-confidence labels), hold (curated validation set), and test subsets. The training pipeline was updated to support multi-subset loading, with tier1 and tier3 used for training and hold split and used for the validation set (50% of hold) and testing set (50% of hold). After applying the damage_only image filter, the effective training pool comprised approximately 2,800 source image pairs from the combined tier1 and tier3 subsets, with 223 samples in the hold validation set, and 224 samples in the hold testing set.

## 5.2 Pretrained Encoder

The from-scratch U-Net encoder was replaced with a ResNet34 encoder pretrained on ImageNet, implemented via the segmentation-models-pytorch library. This modification provides substantially better initial feature representations: the encoder enters training already capable of detecting edges, textures, and structural forms, rather than beginning from random initialisation. The decoder architecture and training loop were unchanged. The library automatically adapts the pretrained encoder to accept 6-channel inputs while preserving pretrained feature representations.

The impact of this change was immediate and significant. Under the previous from-scratch architecture, meaningful predictions first appeared around epoch 20 and the model required approximately 50 epochs to reach a validation IoU of 0.46. With the pretrained encoder, an IoU of 0.44 was achieved by epoch 5, and peak performance of 0.52 was observed at epoch 27 within a 30-epoch run. Training throughput also improved from approximately 47 seconds per epoch to 29 seconds per epoch on the same hardware. This represented the single largest observed improvement in convergence speed and validation performance during development.

## 5.3 Google Earth Domain Data

The 23 manually-collected Google Earth image pairs were normalised to 1024x1024 resolution and reformatted to match the xBD naming and mask encoding convention. Mask pixel values were remapped from the LabelMe output convention (value 1 for damage) to the xBD convention (value 4 for major damage) using nearest-neighbour resampling to preserve label integrity. These samples are used alongside xBD for training.

# 6. Current Status & Results

## 6.1 Training Performance

The most recent completed training run covered 60 epochs on the full xBD damage-filtered training set using the pretrained ResNet34 encoder. Selected epoch results are summarised below:

**Epoch 1:** Loss: 1.3897 | Val IoU: 0.1973 | LR: 1.00e-04

**Epoch 5:** Loss: 0.9361 | Val IoU: 0.4374 | LR: 1.00e-04

**Epoch 27:** Loss: 0.6056 | Val IoU: 0.5814 | LR: 5.00e-05

**Epoch 38:** Loss: 0.5260 | Val IoU: 0.6073 | LR: 2.50e-05 (Peak Val IoU)

**Epoch 60:** Loss: 0.4635 | Val IoU: 0.5775 | LR: 3.13e-06

Qualitative inspection using an internal visualisation pipeline confirms that the model produces spatially coherent predictions that correctly identify damaged building footprints in many cases, particularly at a probability threshold of 0.2-0.3. These results indicate that the model is learning meaningful spatial representations of collapsed structures despite significant domain variability, label noise, and severe foreground-background imbalance within the dataset.

## 6.2 Evaluation Metrics

The primary evaluation metric is Intersection over Union (IoU) computed at a threshold of 0.5 on the binary prediction mask.  IoU is preferred over pixel accuracy due to the severe class imbalance in the dataset: a model predicting all background achieves >90% pixel accuracy but zero IoU. While quantitative evaluation uses a fixed threshold of 0.5, lower thresholds (0.2–0.3) as well as higher thresholds (0.8-0.9) are sometimes used during qualitative inspection to visualise lower-confidence spatial predictions. The combined BCE+Dice loss formulation aligns training directly with region-overlap performance.

The model was evaluated on the unseen test split (224 images) using both per-image and global confusion matrix styles. Despite the severe class imbalance, the model achieved high spatial agreement:

    Per-image Avg IoU: 0.5883

    Global IoU: 0.6113

    Precision: 0.7198

    Recall: 0.8023

    F1-Score: 0.7588

    Overall Accuracy: 97.10%

The high Recall (80.2%) confirms the model's effectiveness in identifying damaged structures, while the Precision (71.9%) demonstrates a significant reduction in false positives compared to early iterations. The convergence of Global and Per-image IoU suggests consistent performance across varying scales of urban damage.

# 7. Technical Stack Summary

**Language:** Python 3.x

**Deep Learning Framework:** PyTorch

**Segmentation Library:** segmentation-models-pytorch (smp)

**Model Architecture:** U-Net with pretrained ResNet34 encoder (ImageNet weights)

**Input Format:** 6-channel tensor: pre-event RGB + post-event RGB, stacked channel-wise

**Output Format:** Single-channel binary logit mask, (B, 1, H, W)

**Loss Function:** BCEWithLogitsLoss (pos_weight=10.0) + Dice Loss

**Optimiser:** Adam, lr=1e-4, ReduceLROnPlateau scheduler (factor=0.5, patience=5)

**Training Hardware:** NVIDIA RTX 4050 Laptop GPU, 6GB VRAM (CUDA)

**Primary Dataset:** xBD (xView2) + manually collected Google Earth imagery — tier1 + tier3 subsets, damage_only filtered (2800 samples)

**Validation Dataset:** 50% of xBD hold subset, damage_only filtered (223 samples)

**Testing Dataset:** 50% of xBD hold subset, damage_only filtered (224 samples)

**Crop Size:** 256x256 pixels from 1024x1024 source images

**Batch Size:** 8

---

*— End of Report —*