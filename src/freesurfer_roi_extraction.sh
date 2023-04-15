#!/bin/bash

# mri_extract_label ./seg_subjs/subj00/mri/aseg.mgz 17 ./study/subjects/subj00/masks/ subj00_freesurfer_mask_hippocampus_left.nii.gz -exit_none_found
# 
# mri_extract_label ./seg_subjs/subj00/mri/aseg.mgz 18 ./study/subjects/subj00/masks/subj00_freesurfer_mask_amygdala_left.nii.gz
# 
# mri_extract_label ./seg_subjs/subj00/mri/aseg.mgz 53 ./study/subjects/subj00/masks/subj00_freesurfer_mask_hippocampus_right.nii.gz
# 
# mri_extract_label ./seg_subjs/subj00/mri/aseg.mgz 54 ./study/subjects/subj00/masks/subj00_freesurfer_mask_amygdala_right.nii.gz

## STRIA TERMINALIS

# (LEFT) AMYGDALA -> BNST
# tckgen -nthreads 4 -algorithm iFOD2 -select 1000 -seeds 1M -seed_image ./stria_termialis/T1/subj00_aseg_Left-Amygdala.nii.gz -seed_unidirectional -include_ordered ./fornix/FA/subj00_FSL_HCP1065_FA_1mm_mask_fornixST_left.nii.gz -include_ordered ./stria_termialis/T1/subj00_FSL_HCP1065_FA_1mm_mask_BNST_left.nii.gz -exclude ./fornix/T1/subj00_aseg_Left-Hippocampus.nii.gz -exclude ./fornix/FA/subj00_FSL_HCP1065_FA_1mm_mask_fornix.nii.gz -stop -fslgrad ../dMRI/preproc/subj00_dmri_preproc.bvec ../dMRI/preproc/subj00_dmri_preproc.bval ../dMRI/ODF/MSMT-CSD/subj00_MSMT-CSD_WM_ODF.nii.gz ../dMRI/tractography/STLeftGo.tck

# (RIGHT) AMYGDALA -> BNST
# tckgen -nthreads 4 -algorithm iFOD2 -select 1000 -seeds 1M -seed_image ./stria_termialis/T1/subj00_aseg_Right-Amygdala.nii.gz -seed_unidirectional -include_ordered ./fornix/FA/subj00_FSL_HCP1065_FA_1mm_mask_fornixST_right.nii.gz -include_ordered ./stria_termialis/T1/subj00_FSL_HCP1065_FA_1mm_mask_BNST_right.nii.gz -exclude ./fornix/T1/subj00_aseg_Right-Hippocampus.nii.gz -exclude ./fornix/FA/subj00_FSL_HCP1065_FA_1mm_mask_fornix.nii.gz -stop -fslgrad ../dMRI/preproc/subj00_dmri_preproc.bvec ../dMRI/preproc/subj00_dmri_preproc.bval ../dMRI/ODF/MSMT-CSD/subj00_MSMT-CSD_WM_ODF.nii.gz ../dMRI/tractography/STRightGo.tck

# (LEFT) BNST -> AMYGDALA
# tckgen -nthreads 4 -algorithm iFOD2 -select 1000 -seeds 1M -seed_image subj00_template_FSL_HCP1065_FA_1mm_mask_BNST_left.nii.gz -seed_unidirectional -include_ordered subj00_template_FSL_HCP1065_FA_1mm_mask_fornixST_left.nii.gz -include_ordered subj00_freesurfer_mask_amygdala_left.nii.gz -exclude subj00_freesurfer_mask_hippocampus_left.nii.gz -exclude subj00_freesurfer_mask_fornix.nii.gz -exclude subj00_template_FSL_HCP1065_FA_1mm_mask_fornix.nii.gz -stop -fslgrad subj00_dmri_preproc.bvec subj00_dmri_preproc.bval subj00_MSMT-CSD_WM_ODF.nii.gz STLeftReturn.tck

# (RIGHT) BNST -> AMYGDALA
# tckgen -nthreads 4 -algorithm iFOD2 -select 1000 -seeds 1M -seed_image subj00_template_FSL_HCP1065_FA_1mm_mask_BNST_right.nii.gz -seed_unidirectional -include_ordered subj00_template_FSL_HCP1065_FA_1mm_mask_fornixST_right.nii.gz -include_ordered subj00_freesurfer_mask_amygdala_right.nii.gz -exclude subj00_freesurfer_mask_fornix.nii.gz -exclude subj00_freesurfer_mask_hippocampus_right.nii.gz -exclude subj00_template_FSL_HCP1065_FA_1mm_mask_fornix.nii.gz -stop -fslgrad subj00_dmri_preproc.bvec subj00_dmri_preproc.bval subj00_MSMT-CSD_WM_ODF.nii.gz STRightReturn.tck

## FORNIX TODO

# (LEFT) HIPPOCAMPUS -> MAMMILLARY
# tckgen -nthreads 4 -algorithm iFOD2 -select 1000 -seeds 1M -seed_image ./fornix/T1/subj00_aseg_Left-Hippocampus.nii.gz -seed_unidirectional -include_ordered ./fornix/FA/subj00_FSL_HCP1065_FA_1mm_mask_fornixST_left.nii.gz -include_ordered ./fornix/FA/subj00_FSL_HCP1065_FA_1mm_mask_fornix.nii.gz -exclude ./stria_termialis/T1/subj00_aseg_Left-Amygdala.nii.gz -exclude ./stria_termialis/T1/subj00_FSL_HCP1065_FA_1mm_mask_BNST_left.nii.gz -stop -fslgrad ../dMRI/preproc/subj00_dmri_preproc.bvec ../dMRI/preproc/subj00_dmri_preproc.bval ../dMRI/ODF/MSMT-CSD/subj00_MSMT-CSD_WM_ODF.nii.gz ../dMRI/tractography/FornixLeftGo.tck

# (RIGHT) HIPPOCAMPUS -> MAMMILLARY
# tckgen -nthreads 4 -algorithm iFOD2 -select 1000 -seeds 1M -seed_image ./fornix/T1/subj00_aseg_Right-Hippocampus.nii.gz -seed_unidirectional -include_ordered ./fornix/FA/subj00_FSL_HCP1065_FA_1mm_mask_fornixST_right.nii.gz -include_ordered ./fornix/FA/subj00_FSL_HCP1065_FA_1mm_mask_fornix.nii.gz -exclude ./stria_termialis/T1/subj00_aseg_Right-Amygdala.nii.gz -exclude ./stria_termialis/T1/subj00_FSL_HCP1065_FA_1mm_mask_BNST_right.nii.gz -stop -fslgrad ../dMRI/preproc/subj00_dmri_preproc.bvec ../dMRI/preproc/subj00_dmri_preproc.bval ../dMRI/ODF/MSMT-CSD/subj00_MSMT-CSD_WM_ODF.nii.gz ../dMRI/tractography/FornixRightGo.tck

## THALAMOCORTICAL

# (LEFT) THALAMUS -> CINGULATE ANTERIOR CORTEX
# tckgen -nthreads 4 -algorithm iFOD2 -select 1000 -seeds 1M -seed_image ./cortex/T1/subj00_aseg_Left-Thalamus.nii.gz -seed_unidirectional -include ./cortex/T1/subj00_aparc+aseg_ctx-lh-caudalanteriorcingulate.nii.gz -include ./cortex/T1/subj00_aparc+aseg_ctx-lh-rostralanteriorcingulate.nii.gz -stop -fslgrad ../dMRI/preproc/subj00_dmri_preproc.bvec ../dMRI/preproc/subj00_dmri_preproc.bval ../dMRI/ODF/MSMT-CSD/subj00_MSMT-CSD_WM_ODF.nii.gz ../dMRI/tractography/CortACCLeftGo.tck

# (RIGHT) THALAMUS -> CINGULATE ANTERIOR CORTEX
# tckgen -nthreads 4 -algorithm iFOD2 -select 1000 -seeds 1M -seed_image ./cortex/T1/subj00_aseg_Right-Thalamus.nii.gz -seed_unidirectional -include ./cortex/T1/subj00_aparc+aseg_ctx-rh-caudalanteriorcingulate.nii.gz -include ./cortex/T1/subj00_aparc+aseg_ctx-rh-rostralanteriorcingulate.nii.gz -stop -fslgrad ../dMRI/preproc/subj00_dmri_preproc.bvec ../dMRI/preproc/subj00_dmri_preproc.bval ../dMRI/ODF/MSMT-CSD/subj00_MSMT-CSD_WM_ODF.nii.gz ../dMRI/tractography/CortACCRightGo.tck

# (LEFT) THALAMUS -> INSULA
# tckgen -nthreads 4 -algorithm iFOD2 -select 1000 -seeds 1M -seed_image ./cortex/T1/subj00_aseg_Left-Thalamus.nii.gz -seed_unidirectional -include ./cortex/T1/subj00_aparc+aseg_ctx-lh-insula.nii.gz -stop -fslgrad ../dMRI/preproc/subj00_dmri_preproc.bvec ../dMRI/preproc/subj00_dmri_preproc.bval ../dMRI/ODF/MSMT-CSD/subj00_MSMT-CSD_WM_ODF.nii.gz ../dMRI/tractography/CortInsulaLeftGo.tck

# (RIGHT) THALAMUS -> INSULA
# tckgen -nthreads 4 -algorithm iFOD2 -select 1000 -seeds 1M -seed_image ./cortex/T1/subj00_aseg_Right-Thalamus.nii.gz -seed_unidirectional -include ./cortex/T1/subj00_aparc+aseg_ctx-rh-insula.nii.gz -stop -fslgrad ../dMRI/preproc/subj00_dmri_preproc.bvec ../dMRI/preproc/subj00_dmri_preproc.bval ../dMRI/ODF/MSMT-CSD/subj00_MSMT-CSD_WM_ODF.nii.gz ../dMRI/tractography/CortInsulaRightGo.tck

usage()
{
  echo "Usage:      [ -s | --subject <IdSubject> ]
            [ -o | --output <Path> ] 
            [ -i | --interval]
            <SegName0> .. <SegNameN> | <StartSeg> <EndSeg> (if in -interval mode)"
  exit 2
}

# "-o abc:d:" means "short options abcd, of which c: and d: require a value."
# "-l alpha,bravo,charlie:,delta:" means "alpha and bravo require no values passing, but "charlie:" and "delta:" do"
# "-a" Allow long options to start with a single '-'.
# "-n prog_name" The name that will be used by the getopt(3) routines when it reports errors

parsed_args=$(getopt -a -n tractography -o :s:o:i -l subject:,output:,interval -- "$@")
valid_args=$? # check if getopt didn't failed
if [ "$valid_args" != "0" ]; then
  usage
fi
echo "Parsed args is ${parsed_args}"

eval set -- "$parsed_args" # sets positional parameters for a command that will be executed by eval1.

# initialization
SUBJ=""
output_dir=""
interval=0

while :
do
  case "$1" in
    -s | --subject)   SUBJ="${2}"       ; shift 2  ;;
    -o | --output)    output_dir="${2}" ; shift 2  ;;
    -i | --interval)  interval=1        ; shift 1  ;;

    # -- means the end of the arguments; drop this, and break out of the while loop
    --) shift; break ;;

    # If invalid options were passed, then getopt should have reported an error,
    # which we checked as VALID_ARGUMENTS when getopt was called...
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done

if [ $interval -eq 1 ]
then
    if [ $# -ne 2 ]
    then
        echo "Interval options needs <StartSeg> <EndSeg>"
        exit 2
    fi
    
    segs=""
    for (( seg="${1}"; seg<="${2}"; seg+=1))
    do
        segs+="${seg} "
    done

    eval set -- "$segs"
fi

if [ $# -eq 0 ]
then
    echo "At least one Segment to extract"
    exit 2
fi

mri_path="$SUBJECTS_DIR/$SUBJ/mri"

if [ ! -d $mri_path ]
then
    echo "The subject doesn't exist"
    exit 1
fi

if [ ! -d $output_dir ]
then
    echo "The output dir doesn't exist"
    exit 1
fi

declare -a types=("aseg" "aparc+aseg") # "aparc.a2009s+aseg" "fornix")

## TODO forse era meglio farla in python questa qua
for seg in $*
do
    
    # get the name of the segment
    ## TODO ma se vuoi implementarla in python devi capire come scrivere questo qua
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
