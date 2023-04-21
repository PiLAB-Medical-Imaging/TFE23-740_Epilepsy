#!/bin/bash
#
#SBATCH --job-name=TRACKING
#SBATCH --ntask=23
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=24:00:00
#SBATCH --mail-user=michele.cerra@student.uclouvain.be
#SBATCH --mail-type=FAIL
#SBATCH --output=trk_out.txt
#SBATCH --error=trk_err.txt

module load freesurfer

PROJECT_DIR=$HOME/Epilepsy-dMRI-VNS # It works only in a working directory in which the project is in Home
export SUBJECTS_DIR=$PROJECT_DIR/seg_subjs

mkdir $PROJECT_DIR/seg_subjs
mkdir $PROJECT_DIR/src/outputs
rm $PROJECT_DIR/src/outputs/*

srun python3 $PROJECT_DIR/src/tractography.py