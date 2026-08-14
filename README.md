# Collapse & Damage Detection Model
## Technical Progress Report

**August 2026**

---

**Project:** Satellite-based building damage segmentation and rubble volume quantification for conflict and disaster zones

**Repository:** github.com/S-Banna/CnDModel

**Status:** Active Development — Model training ongoing

**Report Period:** October 2025 – August 2026

---

# 1. Project Overview

This project develops an automated satellite-imagery analysis pipeline for rapid assessment of structural damage and construction-and-demolition (C&D) debris following disasters and armed conflicts. The broader objective is to support scalable post-conflict recovery and humanitarian response efforts by reducing reliance on slow, manual, and resource-intensive damage assessment methodologies. Conventional field-based assessment workflows are often difficult to conduct rapidly or safely in heavily damaged or conflict-affected environments, particularly at regional scale.

The current phase focuses on deep learning-based pixel-level segmentation of collapsed or heavily damaged structures from paired pre-event and post-event satellite imagery. These segmentation outputs form the foundation for a planned second-stage debris quantification pipeline intended to estimate rubble volume, material composition, and reconstruction burden at building and neighbourhood scales.

## 1.1 Overview of the Proposed Framework

The proposed framework performs binary semantic segmentation of satellite imagery to identify collapsed or heavily damaged buildings following disasters or armed conflicts. The model forms the first stage of a broader damage assessment pipeline, producing building-level damage masks that are subsequently used for rubble quantification and reconstruction analysis. While the following sections describe each component in detail, this section provides a high-level overview of the overall methodology.

Training is primarily based on the xView2 Building Damage Assessment (xBD) dataset, a large-scale public benchmark widely used throughout the remote sensing literature for post-disaster damage assessment. The dataset contains paired pre-event and post-event high-resolution satellite imagery together with expert-annotated building damage labels spanning multiple disaster types worldwide. In addition, a smaller collection of manually acquired Google Earth image pairs was incorporated to begin adapting the model toward the project's target domain. Further details regarding both datasets and their preprocessing are presented in Sections 2 and 5.

Unlike the original xBD benchmark, which formulates damage assessment as a building localisation followed by four-class damage classification (no damage, minor damage, major damage, and destroyed), this work simplifies the task into a binary segmentation problem. During preprocessing, the original xBD labels corresponding to major damage and destroyed (classes 3 and 4) are merged into a single positive damage class, while all remaining pixels are treated as background. This formulation was selected because the downstream objective of the project is rubble detection and volume estimation, where accurately identifying severely damaged structures is more important than distinguishing between intermediate damage categories.

The segmentation network is implemented as a U-Net using a pretrained ResNet34 encoder provided by the segmentation-models-pytorch library. Paired pre-event and post-event RGB images are stacked channel-wise into a six-channel input tensor, allowing the network to learn spatial and temporal changes jointly. The model produces a dense pixel-level probability map indicating the likelihood that each pixel belongs to a heavily damaged or collapsed structure, which is converted into a binary damage mask using a configurable probability threshold (see Section 6.1). Rather than manually designing the encoder-decoder architecture, the project adopts the standard pretrained ResNet34 encoder together with its corresponding U-Net decoder and skip connections. Model optimisation is performed end-to-end using a combined Binary Cross-Entropy and Dice loss. The architecture, training procedure, and evaluation methodology are discussed in detail throughout Sections 3–6.

# 2. Phase 1: Exploration & Prototyping (Oct – Dec 2025)

The project was initiated in October 2025 with an exploratory phase focused on understanding the problem space and establishing a working data pipeline. Early work consisted of classical image processing approaches, repository organisation, and initial data collection efforts.

## 2.1 Classical Change Detection Experiments

Initial experiments employed simple pixel-differencing between pre/post image pairs as a baseline change detection approach. Prototype scripts were written to compare image pairs and visualise change heatmaps. While these methods confirmed that meaningful signal exists between pre/post pairs, they demonstrated fundamental limitations: sensitivity to lighting variation, seasonal change, and camera angle differences caused high false positive rates on non-damaged regions. These experiments were archived into a legacy/ directory and the project transitioned to a supervised deep learning approach.

## 2.2 Initial Data Collection & Baseline Datasets

Early data collection encountered significant obstacles. Access to commercial satellite data providers (including Maxar and NASA portals) was either cost-prohibitive or operationally restricted. As an interim measure, 23 image pairs were manually collected from Google Earth, capturing pre/post views of conflict-affected urban areas in the target region. These images served as the first domain-specific training resource. 

To expand the training pool to a viable scale for deep learning, the pipeline was integrated with the **xBD (xView2 Building Damage Assessment) dataset**. The xBD dataset is a massive, publicly available benchmark specifically selected for this project due to its scale and diversity. It encompasses over 11,000 high-resolution, paired pre- and post-disaster satellite imagery pairs spanning 19 major disaster events worldwide. These events, dating across multiple recent years, cover diverse environmental settings and a wide variety of structural hazards, including earthquakes, hurricanes, floods, volcanic eruptions, and wildfires. This broad representation of disaster typologies allows models to overcome context-specific biases and learn robust features generalizable to novel conflict zones and disaster areas. The dataset includes pixel-level ground truth annotations derived from expert assessments of building polygons, utilizing a standardized grading standard to measure structural impacts *(elaborated further in Section 5.1)*. A separate holdout domain dataset for cross-domain evaluation is pending transfer from a collaborating researcher.

## 2.3 Labeling Protocols & Data Preprocessing

To integrate the heterogeneous sources of satellite imagery, specifically the hand-collected Google Earth imagery and the standardized xBD repository, a rigorous data processing and annotation pipeline was developed.

### 2.3.1 Manual Annotation & Crowd-Sourced Labeling
For the 23 manually collected Google Earth image pairs, regional structural damage was annotated using the LabelMe polygon annotation tool. Structural collapses and heavily damaged footprints were identified as the primary target instances. Each damaged site was manually traced to capture the exact geometric shape of the rubble fields, generating localized vector inputs that complement the larger xBD dataset.

### 2.3.2 Preprocessing & Mask Categorization
Because raw annotations from tools like LabelMe default to standard RGB color-coded vectors, a multi-step preprocessing sequence was designed to homogenize the inputs for PyTorch dataset injection:
- **Resolution Normalization:** Image pairs were resampled to uniform dimensions matching those of the xBD dataset (1024x1024 resolution).
- **Label Mapping and Mask Encoding:** Vector polygons were converted into discrete integer-valued grayscale label masks. To maintain absolute compatibility with the xBD label convention, background pixels were mapped to an integer value of `0`, while structural damage and rubble zones were mapped to a fixed integer value of `4` (the xBD equivalent for total collapse). This unified binary masking scheme ensures a stable loss signal during training.

## 2.4 Repository & Infrastructure Setup

Concurrent with data collection, the repository was restructured for collaborative development. A configuration system using `config.yaml` was introduced to decouple data paths from code, enabling the pipeline to run across different machines without code modification. Large data files and model weights were excluded from version control via `.gitignore`. The codebase was separated into `src/` (model and training code) and `data/` (configuration and local data references).

# 3. Phase 2: Architecture & Dataset Selection (Jan – Feb 2026)

## 3.1 Architecture Decision: U-Net

Following the classical approach being discarded, the project adopted a U-Net encoder-decoder convolutional architecture for semantic segmentation. The U-Net was selected on the basis of several properties well-suited to this task:

- Skip connections between encoder and decoder preserve spatial detail lost during downsampling, which is critical for accurate building boundary delineation at pixel level.
- The architecture is well-established for binary segmentation tasks on satellite and aerial imagery, with strong prior literature supporting its efficacy.
- It accepts arbitrary-channel inputs, making the 6-channel stacked pre/post tensor (2 × RGB) a natural fit without architectural modification.
- It is computationally tractable on consumer GPU hardware, enabling rapid iteration and practical experimentation under limited compute constraints.

The initial implementation comprised a three-level encoder (64, 128, 256 channels), a 512-channel bottleneck, and a symmetric decoder with transposed convolution upsampling and skip connections. All weights were randomly initialised and trained from scratch in the initial experiments.

This initial architecture served as a development baseline and was later replaced by the pretrained ResNet34 U-Net described in Section 5.2, which constitutes the current model used throughout the remainder of this report.

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

The full xBD dataset (~11,000 image pairs across all disaster types) was obtained. The dataset is partitioned into tier1 (high-confidence labels), tier3 (lower-confidence labels), hold (curated validation set), and test subsets. The training pipeline was updated to support multi-subset loading, with tier1 and tier3 used for training, and hold used for the validation set (50% of hold) and testing set (50% of hold). The xBD dataset was used as-is with a `damage_only` filter applied, meaning only images which contain at least one damaged/collapsed building were used. After applying the `damage_only` image filter, the effective training pool comprised approximately 2,800 source image pairs from the combined tier1 and tier3 subsets, with 223 samples in the hold validation set, and 224 samples in the hold testing set.

## 5.2 Pretrained Encoder

During the model architecture design stage, the initial from-scratch U-Net encoder described in Section 3.1 was replaced with a pretrained ResNet34 encoder before training. This modification was introduced to leverage transfer learning, improving feature extraction and accelerating convergence while retaining the original U-Net decoder and skip connections.

The initial from-scratch U-Net implementation served as a development baseline for validating the data pipeline, loss formulation, and overall training procedure. Once the training pipeline had been verified, the encoder was replaced with a pretrained ResNet34 backbone to investigate whether transfer learning could improve convergence speed and segmentation performance. This architecture constitutes the final model used throughout the remainder of this work.

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

Qualitative inspection using an internal visualisation pipeline confirms that the model produces spatially coherent predictions that correctly identify damaged building footprints in many cases, particularly at a probability threshold* of 0.2-0.3. These results indicate that the model is learning meaningful spatial representations of collapsed structures despite significant domain variability, label noise, and severe foreground-background imbalance within the dataset.

***Threshold Definition:** The model outputs a continuous logit value for each pixel, which is converted to a probability through a sigmoid activation function. Binary segmentation masks are obtained by applying a threshold to these probabilities, where pixels with predicted probabilities exceeding the threshold are classified as damaged and all remaining pixels are classified as background. Unless otherwise stated, a threshold of 0.5 is used for all quantitative evaluation metrics reported. For qualitative visualization purposes, alternative thresholds may be applied to inspect lower- or higher-confidence predictions and assess the spatial distribution of model confidence.

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

**Primary Dataset:** xBD (xView2) + manually collected Google Earth imagery

**Dataset Partitioning:** The xBD dataset was provided with predefined subsets: tier1, tier3, and hold. All tier1 and tier3 samples were used exclusively for training. The predefined hold subset was reserved for evaluation and further divided into validation and testing sets using a fixed 50/50 random split.

**Training Dataset:** xBD tier1 and tier3 subsets + manually collected Google Earth imagery, damage_only filtered (2800 samples)

**Validation Dataset:** 50% of xBD hold subset, damage_only filtered (223 samples)

**Testing Dataset:** 50% of xBD hold subset, damage_only filtered (224 samples)

**Crop Size:** 256x256 pixels from 1024x1024 source images

**Batch Size:** 8

# 8. Rubble Quantification Pipeline

## 8.1 Overview

The rubble quantification pipeline forms the second stage of the proposed framework. While the damage segmentation model identifies collapsed or heavily damaged buildings from paired satellite imagery, this stage estimates the physical consequences of that damage, including rubble volume, material composition, and approximate cleanup requirements.

The pipeline is designed as a modular framework in which the segmentation outputs generated during Phase 1 are processed individually for each detected building. Using the detected building footprint together with metadata such as the ground sampling distance (GSD) and structure height, the framework estimates a range of engineering quantities that may assist post-conflict damage assessment and reconstruction planning.

Unlike the segmentation stage, which is entirely data-driven, this phase combines image-derived measurements with engineering assumptions regarding typical building characteristics and construction materials.

## 8.2 Pipeline Inputs

Each sample is defined through a configuration entry containing the required information for processing.

| Parameter                         | Description                                       |
|-----------------------------------|---------------------------------------------------|
| Pre image                         | Pre-conflict satellite image                      |
| Post image                        | Post-conflict satellite image                     |
| Ground-truth mask (optional)      | Used only for evaluation metrics                  |
| Ground Sampling Distance (GSD)    | Spatial resolution in metres per pixel            |
| Structure type                    | Manually selected during inspection               |

The segmentation model generates a binary damage mask which serves as the primary input to the rubble estimation stage.

## 8.3 Processing Pipeline

For each image pair, the pipeline performs the following operations:

1. Load the pretrained segmentation model.
2. Predict a binary damage mask from the paired pre- and post-conflict imagery.
3. Optionally compare the prediction against a manually annotated ground-truth mask and compute segmentation performance metrics.
4. Extract individual damaged buildings using connected-component analysis.
5. Estimate the physical properties of each detected building.
6. Export visualisations, tabulated results, and summary statistics.

This workflow enables multiple image pairs to be processed sequentially using a batch configuration file.

## 8.4 Building Extraction

The predicted binary damage mask is first refined using morphological closing to remove small gaps and connect neighbouring damaged regions.

Connected-component analysis is then applied to identify individual damaged structures. Small detections below a predefined pixel-area threshold are discarded to reduce noise and false detections.

Each remaining connected component is treated as an individual damaged building and assigned a unique identifier for subsequent analysis.

## 8.5 Rubble Quantification

For each detected building, the framework estimates:

- Building footprint area
- Building height
- Building volume
- Rubble volume
- Material composition
- Estimated cleanup requirements

The footprint area is computed directly from the segmented building mask using the supplied ground sampling distance. Building height is currently estimated using engineering assumptions associated with the selected structure type, although the framework is designed to accommodate externally generated height estimates in the future, using techniques such as shadow height estimation or photogrammetric reconstruction (e.g., MicMac, an open-source photogrammetry package).

Using these geometric properties, the pipeline estimates rubble volume before approximating the mass of major construction materials, including concrete, steel, masonry, wood, and other materials. Finally, approximate cleanup durations are estimated using simplified productivity assumptions for manual labour and construction equipment.

## 8.6 Pipeline Outputs

For every processed sample, the pipeline automatically generates an output directory containing:

| Output                        | Description                                           |
|-------------------------------|-------------------------------------------------------|
| Pipeline visualisation        | Summary figure showing the segmentation results       |
| Building identification map   | Damaged buildings labelled with unique IDs            |
| Rubble mass table             | Estimated volume and material quantities              |
| Cleanup table                 | Estimated cleanup effort                              |
| Accuracy report               | Segmentation metrics (when ground truth is available) |
| Summary report                | Aggregate statistics for the processed scene          |

These outputs provide both qualitative visualisations and quantitative estimates that can be incorporated into downstream reconstruction planning or further engineering analysis.

## 8.7 Quantification Methodology

Once individual damaged buildings have been isolated through connected-component analysis (Section 8.4), each detected building is passed through a dedicated quantification routine that converts pixel-based measurements into physical engineering estimates.

**Geometric conversion.** The pixel area of a detected building is first converted into a real-world footprint area using the ground sampling distance supplied for that image pair (area in square metres equals pixel count multiplied by the square of the GSD, since GSD expresses the linear ground distance represented by one pixel: A_footprint = N_pixels × GSD²). Because satellite imagery only captures a building's footprint, not its full volumetric extent, the framework applies a structure-type lookup to estimate the number of floors and, from that, an approximate building height (based on a fixed per-floor height of 3 metres). Multiplying the footprint by the number of floors gives an approximate total built-up area, representing the cumulative floor area across all levels of the structure, not just what is visible from above.

*Area_built = Area_footprint × N_floors*

**Rubble volume estimation.** The built-up area is converted into an estimated rubble volume using a fixed empirical rubble generation rate of 0.8 cubic metres of debris per square metre of built-up area. This coefficient is drawn from prior literature on post-conflict debris estimation (Tamraz, Srour & Chehab, 2012) and reflects observed relationships between structural floor area and resulting debris volume following building collapse. 

*V_rubble = A_built × C_debris*

**Material composition.** Each structure type is associated with a fixed material breakdown (percentage splits for concrete, steel, masonry, wood, and other materials), reflecting typical construction practices for that building category. The total rubble volume is distributed across these material categories proportionally, and each material's volume is then converted to mass using standard reference densities (e.g. 2400 kg/m³ for concrete, 7850 kg/m³ for steel). Summing across all materials yields the total estimated rubble mass for the building.

*m_material = V_rubble × f_material × ρ_material*  &nbsp;&nbsp;&nbsp;*-- where f_material is the fractional composition and ρ_material the reference density*

Currently, all buildings processed in a given batch run share a single user-specified structure type (e.g. "Residential Low Rise"), meaning floor count and material splits are not yet inferred per-building from imagery, but assumed uniformly across a scene. This is a simplification the framework is designed to relax in future work.

## 8.8 Cleanup Effort Estimation

In addition to physical quantities, the pipeline estimates the approximate labour and equipment time required to clear the debris of each building, based on productivity assumptions from the same literature source.

**Manual sorting.** A fraction of the estimated steel tonnage (70%) is assumed to be accessible for manual extraction and sorting, reflecting the practice of manually salvaging steel reinforcement from rubble before mechanical clearance. This accessible steel mass is divided by the combined output of a fixed manual labour crew (4 labourers, each processing 0.7 tonnes per hour) to yield an estimated manual sorting time.

*t_manual = (M_steel × 0.7) / (4 × 0.7 t/hr)*

**Mechanical clearance.** The concrete component of the rubble volume is assumed to be cleared using an excavator with a fixed productivity rate (roughly 70.5 m³/hour, based on a reference machine model), while the total rubble volume as a whole is assumed to require loading and haulage using a wheel loader with its own productivity rate (roughly 160.9 m³/hour, again based on a reference machine). These two operations are modelled independently and their durations are summed alongside the manual sorting time.

*t_excavator = V_concrete / 70.5 m³/hr*

*t_loader = V_total / 160.9 m³/hr*

*t_total = t_manual + t_excavator + t_loader*

**Total effort.** The manual, excavation, and loading hours are summed to produce a total cleanup time per building, which is then converted into estimated workdays assuming an 8-hour working day. These per-building estimates are aggregated across all detected buildings in a scene to produce scene-level totals for rubble mass, volume, and cleanup duration, reported in the final summary output.

## 8.9 End-to-End Pipeline Execution

The full pipeline is orchestrated by a driver script that ties the segmentation model, building extraction, and quantification routine together into a single automated workflow, runnable either on a single image pair or in batch mode across a predefined list of samples.

For each sample, the script first loads the pre- and post-event image pair from disk, converting both to RGB and normalising pixel values to the [0,1] range expected by the model. If available, a corresponding ground-truth damage mask is loaded separately as a single-channel greyscale image, since mask values are treated as class labels rather than colour information. The trained segmentation model is loaded once — with its ResNet34 encoder and decoder weights restored from the saved checkpoint — and reused across all samples in a batch run, avoiding redundant model initialisation and weight loading. The pre- and post-event images are concatenated along the channel axis to form the six-channel input tensor used during training, then reordered into the channel-first format PyTorch expects before being passed through the model in inference mode (with gradient tracking disabled, since no backpropagation is needed at this stage). The raw output logits are passed through a sigmoid function to obtain per-pixel damage probabilities, which are then thresholded (default 0.5, configurable) to produce a binary damage mask.

If a ground-truth mask is supplied, the pipeline computes the same suite of evaluation metrics used during training and testing. This is done by directly comparing the binary prediction against the binarised ground-truth mask on a per-pixel basis to tally true positives, false positives, true negatives, and false negatives, from which precision, recall, F1-score, IoU, and overall accuracy are derived using the same formulas applied to the aggregate test-set evaluation in Section 6.2. Where the ground-truth mask and prediction differ in resolution, the ground truth is resampled using nearest-neighbour interpolation to match the prediction grid before comparison, preserving discrete label values rather than introducing interpolated intermediate ones.

The binary mask is then cleaned using morphological closing, a two-step dilation-then-erosion operation that fills small gaps and holes within damaged regions without materially growing their boundaries — this helps merge fragments of a single collapsed structure that the model may have segmented as disconnected blobs. The size of the structuring element used for this operation is not fixed in pixels; instead it is derived from the scene's GSD so that the closing operation corresponds to a consistent real-world gap distance (approximately 2 metres) regardless of image resolution, ensuring the same physical merging behaviour applies whether the source imagery is coarse or fine-grained. Connected-component labelling is then applied to the cleaned mask, assigning each contiguous group of foreground pixels a unique integer label along with its pixel area and centroid coordinates. Components whose pixel area falls below a minimum threshold are discarded before quantification, filtering out isolated noisy pixels or speckle artefacts that do not correspond to real structures. Each surviving component is quantified using the methodology described in Sections 8.7–8.8.

Finally, the script assembles all outputs for the sample into a dedicated output folder, named automatically from the input filename so that outputs from different scenes do not overwrite one another during batch runs. A composite multi-panel figure is generated showing the pre-image, post-image, damage overlay, ground truth (where available), raw prediction, and border overlay, with the damage and border overlays produced by colour-tinting the post-event image directly wherever the corresponding mask is positive, and the border variant additionally isolating the edge of each damaged region through erosion-based edge extraction so boundaries are visible without obscuring the underlying imagery. A separate labelled building-identification map annotates each detected component with its numeric ID at its centroid, rendered with a dark outline for legibility against varying background intensities. Per-building results are written out as two CSV tables — one covering rubble mass and material breakdown, the other cleanup effort — alongside a plain-text accuracy report (when ground truth is available) and a scene-level summary file totalling rubble volume, mass, and estimated cleanup duration across all detected buildings in the scene.

---

*— End of Report —*