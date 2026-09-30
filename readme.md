# CnDModel

CnDModel is a research project for estimating building damage and construction and demolition (C&D) rubble from pre- and post-disaster satellite imagery.

The project is divided into two phases:

* **Phase 1:** Building-damage segmentation from paired pre- and post-event imagery.
* **Phase 2:** Building-height estimation and rubble quantification.

The detailed research documentation is available in:

* [`docs/docs_P1.md`](docs/docs_P1.md) — Phase 1: damage segmentation and early rubble estimation
* [`docs/docs_P2.md`](docs/docs_P2.md) — Phase 2: height estimation and rubble quantification

## Quick Start

A pretrained Phase 1 model and two sample images are included in the repository so that the inference pipeline can be tested without downloading the training dataset.

### Requirements

The inference pipeline requires Python and the packages listed in the project requirements.

Clone the repository and install the dependencies:

```bash
git clone <repository-url>
cd CnDModel

pip install -r requirements.txt
```

The pretrained model is included as:

```text
model.pth
```

Two sample images are included in:

```text
samples/
```

### Run the sample pipeline

Run:

```bash
python pipeline_plusacc.py --batch
```

This runs the pipeline on the sample images and produces the corresponding output examples.

The pipeline uses the included `model.pth`, so no model training or dataset download is required for this test.

## Project Overview

The current Phase 1 pipeline uses a U-Net segmentation architecture with a ResNet-34 encoder.

The model receives paired pre- and post-event imagery and predicts a pixel-level building-damage mask. The ResNet-34 encoder is initialized with ImageNet-pretrained weights, while the U-Net decoder reconstructs the spatial segmentation output using encoder skip connections.

The broader research pipeline is:

```text
Pre-event image
       +
Post-event image
       ↓
Building damage segmentation
       ↓
Damaged building regions
       ↓
Building height estimation
       ↓
Rubble volume / material estimation
```

The Phase 2 components are currently experimental and are documented separately in [`docs/docs_P2.md`](docs/docs_P2.md).

## Dataset

Phase 1 was primarily developed using the **xBD dataset**, a large paired satellite-image dataset for building damage assessment.

The project uses pre- and post-disaster imagery together with the building-damage annotations provided by xBD.

For the experiments described in the Phase 1 documentation, a subset of approximately 6,000 images was used.

The damage classes used in the training pipeline were combined into a binary damage segmentation target.

See [`docs/docs_P1.md`](docs/docs_P1.md) for the complete preprocessing, training, validation, and experimental details.

## Model

The Phase 1 segmentation model is:

```text
Input: 6 channels
       ├── 3-channel pre-event image
       └── 3-channel post-event image

             ↓

       U-Net / ResNet-34
       pretrained encoder

             ↓

     Binary damage mask
```

The repository contains a trained model for inference:

```text
model.pth
```

The model included in the repository is intended primarily to make the inference pipeline reproducible and immediately testable. It is not a replacement for the training data and experimental setup described in the research documentation.

## Repository Structure

```text
CnDModel/
├── model.pth
├── pipeline-plusacc.py
├── samples/
│   ├── ...
│   └── ...
├── src/
│   ├── segmentation/
│   └── rubble/
├── docs/
│   ├── docs_P1.md
│   └── docs_P2.md
└── requirements.txt
```

The exact contents may change as the project develops.

## Phase 1: Damage Segmentation

Phase 1 focuses on identifying damaged buildings from paired satellite imagery.

The main model is a U-Net with a ResNet-34 encoder. Training used cropped 256×256 image pairs, binary damage masks, weighted BCE + Dice loss, and ImageNet-pretrained encoder weights.

The best experimental validation IoU was approximately 0.6.

More detailed information about:

* data preparation
* model architecture
* training
* loss functions
* hyperparameters
* validation
* inference
* experimental results

is available in [`docs/docs_P1.md`](docs/docs_P1.md).

## Phase 2: Height and Rubble Estimation

Phase 2 investigates how the segmentation output can be extended into quantitative estimates of building height and C&D rubble.

Current approaches under investigation include:

* off-nadir image geometry
* roof/facade segmentation
* Offset-Building Model (OBM)
* existing building-height estimation models
* DSM/elevation data
* photogrammetry

The intended long-term pipeline is to combine building damage, building geometry, and height information to estimate rubble volume and material quantities.

See [`docs/docs_P2.md`](docs/docs_P2.md).

## Research Status

This repository contains an ongoing undergraduate research project. Some components are experimental and should not be interpreted as finalized methods.

Phase 1 provides the current working damage-segmentation pipeline.

Phase 2 is under development and will be evaluated as additional imagery, metadata, and independent height measurements become available.