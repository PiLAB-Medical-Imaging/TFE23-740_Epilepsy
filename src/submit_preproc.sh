#!/bin/bash

# Run preprocess python script in backgroung and redirect the output in out.txt
# preproc.py will create a sbatch job in the ceci cluster and wait the end.

if [[ $# -ne 3 ]]; then
	echo "Invalid number of parameters"
	echo "[name ceci cluster] [path of the folder] [starting_state]"
	exit 1
fi

if [[ ! -d $2 ]]; then
	echo "Error: Directory $2 does not exists."
	exit 1
fi
echo "Directory $2 exists."

python3 preproc.py -w $1 -f $2 -s $3 > out.txt 2>&1 &
jobs

exit 0
