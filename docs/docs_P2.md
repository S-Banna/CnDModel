# Phase 2: Building Height and Rubble Estimation

Phase 2 extends the building-damage segmentation pipeline toward estimating building height to use in the computation of the quantity of construction and demolition (C&D) rubble produced by damaged buildings.

This phase is currently exploratory. The approaches below are being investigated and may be replaced as better data, models, or geometric information become available.

## Things We Are Trying

### 1. Off-nadir imagery geometry

Investigate whether building height can be estimated from the apparent displacement between a building's roof and its ground/facade footprint in off-nadir imagery.

A first-order estimate can be obtained from:

* building pixel displacement
* ground sampling distance (GSD)
* sensor viewing/off-nadir angle

This is intended as an initial geometric baseline rather than a final height-estimation method.

### 2. Roof and facade segmentation

Separate the visible roof from the building facade in oblique imagery.

The current idea is to use the Phase 1 building/damage segmentation to identify building regions, then obtain:

* roof mask
* facade/building mask
* roof-to-ground or roof-to-footprint displacement

These masks can provide the geometric measurements needed for subsequent height estimation.

### 3. OBM

Investigate the Offset-Building Model (OBM) for roof segmentation, building segmentation, and roof-to-footprint offset estimation from oblique aerial/satellite imagery.

OBM produces relative geometric information that may be useful for estimating building height. Metric height estimation will require additional sensor geometry, known heights, or another source of scale.

### 4. Building Height Models

Investigate existing building-height estimation models, including models trained specifically on multi-view or off-nadir remote-sensing imagery.

The main question is whether existing pretrained models can be adapted to the imagery available for this project without requiring an entirely new training dataset.

### 5. DSM / elevation data

Investigate the use of Digital Surface Models (DSM) and related elevation products as an independent source of building height.

This may provide a simpler and more directly measurable height estimate if sufficiently high-resolution elevation data is available for the study areas.

### 6. Photogrammetry

Investigate multi-view photogrammetry as a geometry-based alternative for recovering building height.

Potential tools and approaches include structure-from-motion and multi-view stereo. This requires overlapping imagery with sufficient camera/sensor information and a reliable metric scale.

### 7. Height validation

Collect independent building-height information where possible, such as:

* known building heights
* floor counts
* LiDAR/DSM measurements
* other geospatial elevation products

These measurements will be used to evaluate candidate height-estimation methods.

## Planned Pipeline

The current intended pipeline is:

`Pre/Post imagery → Phase 1 segmentation → Building ROI → Roof/Facade segmentation → Height estimation → Building geometry → C&D rubble estimation`

The exact implementation of the Phase 2 pipeline is not yet finalized.

## Current Status
 
Phase 2 is under active investigation.
 
No height estimate has been validated yet, and nothing has been tested on the target (Lebanon) imagery. Quantitative results so far come from BANDON and a few xBD examples, not from the target imagery.
 
### Status by approach
 
| # | Approach | Status |
|---|----------|--------|
| 1 | Off-nadir geometry | Relationship defined: `h = d · GSD / tan(θ)`, where `d` is the roof-to-footprint displacement in pixels. A 1 px error in `d` gives about `GSD / tan(θ)` metres of height error (roughly 1.1 m at 25° and 0.9 m at 30° for 0.5 m GSD). The displacement estimator is not implemented yet. |
| 2 | Roof/facade segmentation |  SAM 3 prototypes implemented. Zero-shot SAM 3 (building minus roof) was evaluated on BANDON test tiles and is weak on facades; a U-Net trained on BANDON does much better (see BANDON experiment). Not yet run on Lebanon imagery. |
| 3 | OBM | Run on 2 xBD images in WSL; the old MMDetection environment was hard to reproduce and results were poor. This is not a fair test, since xBD is mostly near-nadir and OBM is built for off-nadir imagery. OBM is SAM v1-based and box-prompted, and outputs roof plus roof-to-footprint offset (relative height). It does not segment facades. PolyFootNet weights are public, but its polygon/keypoint code is withheld; roof and offset prediction goes through the OBM code. Not yet run. |
| 4 | DSM / DTM data | Available data was too low quality (more than multiple meters GSD, up to 30m). At high resolutions, a single pixel in the map can represent multiple buildings. |
| 5 | Photogrammetry | Not started. |
| 6 | Height validation | Not started. No reference heights collected yet. |
 
### Roof/facade segmentation prototype
 
* `sam_pipeline.py`: for each Phase 1 building island, take a padded square crop (upscaled to SAM 3's input size), run text prompts for roof and facade, pick the instance that best matches that island, and let the roof win where masks overlap. SAM 3 treats a box prompt as a visual exemplar, not a spatial restriction, so crop-and-select is used instead of box prompting.
* `sam3_roof_building_facade.py`: facade is defined as building minus roof, using "building" and "roof" prompts. The roof is nested inside the building mask and thin edge slivers are removed. Each building gets a status (`ok`, `no_roof`, `no_building`, `no_facade`, `no_roof_no_building`), and masks, an overview image, and a JSON record are saved. The logic was checked with a stubbed model; it has not yet been run on real imagery.
* Manual checks of the direct facade prompt on a few xBD buildings showed some good roof/facade separations, but there is no quantitative evaluation.
### Data
 
* **Pre-disaster (Lebanon):** awaiting delivery. Expected to include off-nadir angle and GSD metadata, off-nadir angle above 25°, and GSD below 1 m. This is not confirmed yet.
* **Post-disaster:** likely Google Earth imagery, manually aligned. It needs no metadata for damage assessment, but it provides no viewing geometry for height. The Phase 1 model may need augmentation or retraining to port to this imagery.
* **Metadata audit:** `metadata_histogram.py` summarizes off-nadir angle, pixel size, scene IDs, and azimuth from label JSONs (xBD layout) and reports the fraction of imagery usable for off-nadir height estimation. It is intended to be rerun on the Lebanon metadata when it arrives.
* **Fallback datasets identified, not yet used:** BANDON has been downloaded and used (U-Net training and evaluation). Not yet used: Maxar Open Data (Beirut 2020), SpaceNet MVOI, BONAI
### Known issues
 
* Adjacent buildings can merge into one Phase 1 connected component, causing SAM 3 to segment only one building or produce inaccurate masks.
* Poor Phase 1 masks degrade the crops and the instance selection. Larger padding and a re-segmentation fallback for incomplete masks still need investigation.
* Facade segmentation is unreliable under occlusion, nearby structures, varying building heights, multiple visible sides, and low off-nadir angles.
* The current xBD examples may not represent the quality or characteristics of the target imagery.
### Next Steps
 
1. Obtain the Lebanon imagery metadata, run the metadata audit, and confirm the off-nadir and GSD constraints.
2. Run both BANDON-trained models on the Lebanon pre-imagery and review the outputs. Hand-label 20–30 buildings for a quantitative check, and fine-tune the U-Net on them if needed. SAM 3 fine-tuning needs a GPU with at least 16 GB.
3. Implement the displacement estimator d and evaluate it against BANDON val offsets.
4. Once target data is available, compare the SAM 3 roof/facade approach with OBM/PolyFootNet on the same buildings against reference heights, and reassess the approach.

### BANDON roof/facade experiment (trained U-Net vs zero-shot SAM 3)

**Dataset.** BANDON is a building change-detection dataset of off-nadir city imagery (2048×2048 tiles, nominal 0.6 m/px, Beijing and Shanghai in-domain, other cities out-of-domain without roof/facade labels). It has no off-nadir angle, GSD, or height metadata. Verified on the local copy:

* `building_labels` PNGs: 0 = background, 1 = roof, 2 = facade (facade = building minus roof; the facade labels are generated by sweeping the roof along its annotated offset).
* Offset labels: 256×256×2 float32 per tile, apparently in full-resolution pixels (range about −67 to +56). On the tile inspected, the stored vectors pointed away from the facade, so the sign convention is the opposite of roof→footprint. `bandon_vis.py` checks the convention against the facade labels automatically.

**Models** (scripts in `src/height/`):

* U-Net: ResNet34 encoder (ImageNet init), 3 classes, 512×512 crops, batch 4 with 4× gradient accumulation, 60 epochs (about 91 min on the laptop GPU). 10% of training locations (all dates together) were held out for checkpoint selection; the best held-out mean IoU (roof, facade) was 0.619.
* SAM 3 zero-shot: 512 px windows upscaled to 1008, prompts "building rooftop"/"roof" and "building", facade = building minus roof. No training.

**Results** (40 BANDON test tiles, same tiles for both models, dataset-level IoU):

| Model | Roof IoU | Facade IoU | Building IoU | Mean IoU (roof, facade) |
|-------|----------|------------|--------------|-------------------------|
| U-Net (trained) | 0.743 | 0.500 | 0.776 | 0.622 |
| SAM 3 (zero-shot) | 0.483 | 0.076 | 0.583 | 0.280 |

**Findings**

* The trained U-Net is far better than zero-shot SAM 3 on BANDON, as expected for a model trained on that distribution. The held-out and test scores agree (0.619 vs 0.622), so the location-grouped split appears clean.
* With this untuned prompting, SAM 3's building-minus-roof facade is unreliable (IoU 0.08). SAM 3 also took about 38 s per tile versus about 0.5 s for the U-Net on the same GPU.
* SAM 3 was not fine-tuned: Meta's training documentation lists 16 GB of VRAM as the minimum, and the available GPU has 6 GB.

**Limitations**

* Scores are on BANDON only (40 tiles, with locations repeated across dates), so they are preliminary. Neither model has been run on Lebanon imagery.
* No roof-offset (`d`) error metric yet. BANDON test has no offsets, but `val` does.
* Model checkpoints are not committed to the repository (see `.gitignore`).