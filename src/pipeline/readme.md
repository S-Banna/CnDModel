**Pipeline**

Performs:

- Damage segmentation using the trained U-Net model.  
- Building extraction from the predicted damage mask.  
- Rubble quantification.  
- Cleanup-time estimation.  
- Export of visualizations and tabulated results.  
  
**Single image**

python pipeline.py 
    --pre pre.png 
    --post post.png 
    --label label.json

The label file is used to automatically obtain GSD (pan_resolution).

**Specify threshold**

Default threshold is 0.5.

python pipeline.py 
    --pre pre.png 
    --post post.png 
    --label label.json 
    --threshold 0.7

**Specify GSD manually**

Overrides values found in the label file.

python pipeline.py 
    --pre pre.png 
    --post post.png 
    --gsd 0.5

**Specify structure type**

Available options:

- Residential Low Rise
- Residential High Rise
- Industrial

Example:

python pipeline.py 
    --pre pre.png 
    --post post.png 
    --label label.json 
    --structure "Industrial"

If omitted:

Structure type not selected.
Using default structure type = Residential Low Rise

**Batch mode**

Edit RUN_LIST in config.py.

Example:

RUN_LIST = [
    {
        "pre": "...",
        "post": "...",
        "mask": "...",
        "label": "...",
        "structure_type": "Residential Low Rise"
    }
]

Run:

python pipeline.py --batch

**GSD handling**

The pipeline determines ground sample distance (GSD) in the following order:

- --gsd
- RUN_LIST["gsd"]
- label.json["metadata"]["pan_resolution"]
- label.json["metadata"]["gsd"] / 4
- Default value: 0.5 m/px

**Output**

For an input image:

z-google-earth_00000020_post_disaster.png

outputs are written to:

pipeline/outputs/z-google-earth_00000020/

**Segmentation figure**

z-google-earth_00000020_pipeline.png  
*Contains:*
Pre-disaster image,
Post-disaster image,
Damage overlay,
Ground truth,
Prediction mask,
Border overlay


**Rubble visualization**

z-google-earth_00000020_rubble.png  
*Contains:*
Connected-component building detections
Building IDs
Numbering consistent with rubble calculations


**Rubble mass table**

rubble_mass.csv  
*Contains:*
Building ID
Area
Built-up area
Rubble volume
Concrete mass
Steel mass
Masonry mass
Wood mass
Other mass
Total mass


**Cleanup table**

rubble_cleanup.csv  
*Contains:*
Building ID
Manual sorting hours
Excavator hours
Loader hours
Total cleanup hours
Estimated workdays