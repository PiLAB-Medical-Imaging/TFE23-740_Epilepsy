mrview study/subjects/$1/dMRI/preproc/${1}_dmri_preproc.nii.gz -interpolation false -overlay.load study/subjects/$1/registration/${1}_T1_brain_reg.nii.gz -overlay.opacity 0.4 -overlay.interpolation false

mrview study/subjects/$1/registration/${1}_T1_brain_reg.nii.gz -interpolation false -overlay.load study/subjects/$1/registration/MNI152_T1_1mm_brain_reg.nii.gz -overlay.opacity 0.4 -overlay.interpolation false -roi.load study/subjects/$1/masks/${1}_Left-Plane1-Mammillary-body_Drawn.nii.gz -roi.load study/subjects/$1/masks/${1}_Right-Plane1-Mammillary-body_Drawn.nii.gz
