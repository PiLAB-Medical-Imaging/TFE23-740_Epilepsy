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

PROJECT_DIR=$HOME/Epilepsy-dMRI-VNS
export SUBJECTS_DIR=$PROJECT_DIR/seg_subjs

SUB_ID=MNI152

srun --cpus-per-task=4 recon-all -all -s $SUB_ID -i $PROJECT_DIR/study/T1/MNI152_T1_05mm.nii.gz -qcache -expert $SUBJECTS_DIR/expert.opts

srun --cpus-per-task=4 mri_cc -force -f -aseg aseg.mgz -o aseg.auto_CCseg.mgz $SUB_ID
