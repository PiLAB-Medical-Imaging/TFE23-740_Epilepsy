#!/bin/bash
#
#SBATCH --job-name=seg
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=michele.cerra@student.uclouvain.be
#
#SBATCH --ntasks=23
#SBATCH --time=0
#SBATCH --time-min=10:0:0
#SBATCH --mem-per-cpu=4000

module load freesurfer

PROJECT_DIR=$HOME/Epilepsy-dMRI-VNS # It works only in a working directory in which the project is in Home
export SUBJECTS_DIR=$PROJECT_DIR/seg_subjs

srun python3 seg_job.py &
jobs

exit 0
