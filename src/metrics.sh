#!/bin/bash

# Run metrics python script in backgroung and redirect the output in out.txt
# metrics.py will create a sbatch job in the ceci cluster and wait the end.

mkdir $PROJECT_DIR/src/outputs
rm $PROJECT_DIR/src/outputs/*

python3 metrics.py -f $1 -m $2 > $PROJECT_DIR/src/outputs/out.txt 2>&1 &
jobs

exit 0