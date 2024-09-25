#!/bin/bash 

# Defin vriables
IMAGE_DIR="data/images/"
OUTPUT_DIR="results/pointcloud/"
FOCAL_LENGTH=2559.68
PRINCIPAL_POINT_X=1536
PRINCIPAL_POINT_Y=1152
RANSAC_ITERATIONS=1000
RANSAC_THRESHOLD=0.01
HARRIS_k=0.04
HARRIS_THRESHOLD=1e6 




#Run the main.py
python3 src/main.py \
    --image_dir "$IMAGE_DIR"\
    --output_dir "$OUTPUT_DIR"\
    --focal_length "$FOCAL_LENGTH"\
    --principal_point "$PRINCIPAL_POINT_X" "$PRINCIPAL_POINT_Y"\
    --k_harris "$HARRIS_k"\
    --threshold_harris "$HARRIS_THRESHOLD"\
    --ransac_iterations "$RANSAC_ITERATIONS"\
    --ransac_threshold "$RANSAC_THRESHOLD"




