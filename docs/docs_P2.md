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

Phase 2 is under active investigation. The immediate objective is to establish a reliable method for estimating building height from the available imagery before attempting full rubble-volume or material-mass estimation.
