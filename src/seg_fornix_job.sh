#!/bin/bash
#
#SBATCH --job-name=seg
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=michele.cerra@student.uclouvain.be
#SBATCH --output=seg_out.txt
#SBATCH --error=seg_err.txt
#
#SBATCH --cpus-per-task=4
#SBATCH --time=0
#SBATCH --time-min=10:0:0
#SBATCH --mem-per-cpu=4000


PROJECT_DIR=$HOME/Epilepsy-dMRI-VNS
export SUBJECTS_DIR=$PROJECT_DIR/seg_subjs

SUB_ID=subj01

srun --cpus-per-task=4 mri_cc -aseg $SUBJECTS_DIR/$SUB_ID/mri/aseg.mgz -o $SUBJECTS_DIR/$SUB_ID/mri/aseg.auto_CCseg.mgz -f -force $SUB_ID
