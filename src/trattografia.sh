#!/bin/bash

# mri_extract_label ./seg_subjs/subj00/mri/aseg.mgz 17 ./study/subjects/subj00/masks/ subj00_freesurfer_mask_hippocampus_left.nii.gz -exit_none_found
# 
# mri_extract_label ./seg_subjs/subj00/mri/aseg.mgz 18 ./study/subjects/subj00/masks/subj00_freesurfer_mask_amygdala_left.nii.gz
# 
# mri_extract_label ./seg_subjs/subj00/mri/aseg.mgz 53 ./study/subjects/subj00/masks/subj00_freesurfer_mask_hippocampus_right.nii.gz
# 
# mri_extract_label ./seg_subjs/subj00/mri/aseg.mgz 54 ./study/subjects/subj00/masks/subj00_freesurfer_mask_amygdala_right.nii.gz
# 
# tckgen -nthreads 4 -algorithm iFOD2 -select 100 -seeds 1M -seed_image subj00_freesurfer_mask_amygdala_left.nii.gz -seed_unidirectional -include_ordered subj00_template_FSL_HCP1065_FA_1mm_mask_fornixST_left.nii.gz -include_ordered subj00_template_FSL_HCP1065_FA_1mm_mask_BNST_left.nii.gz -exclude subj00_freesurfer_mask_hippocampus_left.nii.gz -exclude subj00_template_FSL_HCP1065_FA_1mm_mask_fornix.nii.gz -stop -fslgrad subj00_dmri_preproc.bvec subj00_dmri_preproc.bval subj00_MSMT-CSD_WM_ODF.nii.gz trcActFsl.tck
# 
# tckgen -nthreads 4 -algorithm iFOD2 -select 100 -seeds 1M -seed_image subj00_freesurfer_mask_hippocampus_left.nii.gz -seed_unidirectional -include subj00_template_FSL_HCP1065_FA_1mm_mask_fornix.nii.gz -stop -fslgrad subj00_dmri_preproc.bvec subj00_dmri_preproc.bval subj00_MSMT-CSD_WM_ODF.nii.gz trcActFsl.tck
# 

if [ $# -le 1 ]
then
    echo "invalud num of parameters"
    exit 1
fi

SUBJ=$1
shift

output_dir=$1
shift

mri_path="$SUBJECTS_DIR/$SUBJ/mri"

if [ ! -d $mri_path ]
then
    echo "The subject doesn't exist"
    exit 1
fi

declare -a types=("aseg" "aparc+aseg" "aparc.a2009s+aseg")

for seg in $*
do
    
    # get the name of the segment
    seg_name=$(egrep "^$seg\s+\S+|\n$seg\s+\S+" $FREESURFER_HOME/FreeSurferColorLUT.txt | head -n 1 | tr  "\t" " " | tr -s " " | cut -d " " -f 2)

    found=0

    for type in ${types[@]}
    do
        out_type=$type
        if [ "$type" == "${types[2]}" ]
        then 
            out_type="aparca2009s+aseg"
        fi

        mri_extract_label -exit_none_found ${mri_path}/${type}.mgz $seg ${output_dir}/${SUBJ}_${out_type}_${seg_name}.nii.gz

        # if the label is found exit
        if [ $? -eq 0 ]
        then
            found=1
            break
        else 
            # if it's not found delete the file
            rm ${output_dir}/${SUBJ}_${type}_${seg_name}.nii.gz
        fi
    done

    if [ $found -eq 0 ]
    then
        echo "Label not found for $seg, maybe you have to implement it or doesn't exist"
    fi 
    
done
