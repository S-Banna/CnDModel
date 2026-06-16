**Pipeline**

Single image:

python pipeline.py ^
    --pre pre.png ^
    --post post.png ^
    --label label.json

Optional threshold: (default value is 0.5)

python pipeline.py ^
    --pre pre.png ^
    --post post.png ^
    --label label.json ^
    --threshold 0.7

Batch mode:

Edit RUN_LIST in config.py

python pipeline.py --batch

Output:

pipeline/outputs/
contains both segmentation figure and rubble outputs