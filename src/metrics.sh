#!/bin/bash

# Run metrics python script in backgroung and redirect the output in out.txt
# metrics.py will create a sbatch job in the ceci cluster and wait the end.

if [[ $# -ne 2 ]]; then
	echo "Invalid number of parameters"
	echo "[path of the folder] [model]"
	exit 1
fi

PROJECT_DIR=$HOME/Epilepsy-dMRI-VNS # It works only in a working directory in which the project is in Home

mkdir $PROJECT_DIR/src/outputs
rm $PROJECT_DIR/src/outputs/*

python3 metrics.py -f $1 -m $2 > $PROJECT_DIR/src/outputs/out.txt 2>&1 &
jobs

exit 0
