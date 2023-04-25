#!/bin/bash
#
#SBATCH --job-name=seg
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=michele.cerra@student.uclouvain.be
#SBATCH --output=seg_out.txt
#SBATCH --error=seg_err.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8192

module load freesurfer

PROJECT_DIR=$HOME/Epilepsy-dMRI-VNS # It works only in a working directory in which the project is in Home
export SUBJECTS_DIR=$PROJECT_DIR/seg_subjs

mkdir $PROJECT_DIR/seg_subjs
mkdir $PROJECT_DIR/src/outputs
rm $PROJECT_DIR/src/outputs/*

srun python3 $PROJECT_DIR/src/seg_job.py


