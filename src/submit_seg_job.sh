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

module load freesurfer
export SUBJECTS_DIR=~/seg_subjs

srun --cpus-per-task=4 recon-all -all -s subj00 -i ~/raw/s42428_Sag_T1_MPRAGE_1x1x1_20210814093402_2.nii.gz -T2 ~/raw/s42429_Sag_T2_Cube_1x1x1_20210814093402_3.nii.gz -T2pial -qcache  