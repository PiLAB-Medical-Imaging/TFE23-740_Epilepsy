#!/bin/bash
#
#SBATCH --job-name=TRACKING
#SBATCH --ntask=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2048
#SBATCH --time=24:00:00
#SBATCH --mail-user=michele.cerra@student.uclouvain.be
#SBATCH --mail-type=FAIL
#SBATCH --output=trk_out.txt
#SBATCH --error=trk_err.txt

PROJECT_DIR=$HOME/Epilepsy-dMRI-VNS # It works only in a working directory in which the project is in Home

srun python3 $PROJECT_DIR/src/tractography.py