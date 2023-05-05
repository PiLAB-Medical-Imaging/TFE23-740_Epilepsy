import numpy as np
import nibabel as nib

from nibabel import Nifti1Image

# Prob masks atlas Juelich
mamm_body_map = nib.load("./mammillary_body/Both-Mammillary-body_Juelich.nii.gz")
mamm_body = mamm_body_map.get_fdata()

fornix_juelich = nib.load("./fornix/Both-Fornix_Juelich.nii.gz").get_fdata()

l_cingulum_juelich = nib.load("./cingulum/Left-Cingulum_Juelich.nii.gz").get_fdata()
r_cingulum_juelich = nib.load("./cingulum/Right-Cingulum_Juelich.nii.gz").get_fdata()

# Prob masks atlas Xtract
l_fornix_xtract = nib.load("./fornix/Left-Fornix_Xtract.nii.gz").get_fdata()
r_fornix_xtract = nib.load("./fornix/Right-Fornix_Xtract.nii.gz").get_fdata()

l_cingulum_xtract = nib.load("./cingulum/Left-Cingulum_Xtract.nii.gz").get_fdata()
r_cingulum_xtract = nib.load("./cingulum/Right-Cingulum_Xtract.nii.gz").get_fdata()

# Prob masks  atlas Jhu
l_cingulum_jhu_trac = nib.load("./cingulum/Left-Cingulum_Jhu-tracts.nii.gz").get_fdata()
r_cingulum_jhu_trac = nib.load("./cingulum/Right-Cingulum_Jhu-tracts.nii.gz").get_fdata()

# TODO the density should be thresholded depending on the p-value 
thres = 11

mamm_body[mamm_body < thres] = 0 # threshold the density

# Merging of fornix masks
fornix_masks = [
    np.expand_dims(fornix_juelich, 3),
    np.expand_dims(l_fornix_xtract, 3),
    np.expand_dims(r_fornix_xtract, 3)
    ]
fornix = np.concatenate(fornix_masks, axis=3)
fornix = fornix.mean(axis=3)
fornix[fornix < thres] = 0 # threshold the density 

# Merging of cingulum
l_cingulum_masks = [
    np.expand_dims(l_cingulum_juelich, 3),
    np.expand_dims(l_cingulum_xtract, 3),
    np.expand_dims(l_cingulum_jhu_trac, 3),
]
r_cingulum_masks = [
    np.expand_dims(r_cingulum_juelich, 3),
    np.expand_dims(r_cingulum_xtract, 3),
    np.expand_dims(r_cingulum_jhu_trac, 3),
]
l_cingulum = np.concatenate(l_cingulum_masks, axis=3)
l_cingulum = l_cingulum.mean(axis=3)
l_cingulum[l_cingulum < thres] = 0 # threshold the density

r_cingulum = np.concatenate(r_cingulum_masks, axis=3)
r_cingulum = r_cingulum.mean(axis=3)
r_cingulum[r_cingulum < thres] = 0 # threshold the density

# Saving 
mamm_mask_map = Nifti1Image(mamm_body, mamm_body_map.affine)
nib.save(mamm_mask_map, "./masks/Both-Mammillary-body_Atlas-Merge.nii.gz")

fornix_mask_map = Nifti1Image(fornix, mamm_body_map.affine) # they have se same affine information
nib.save(fornix_mask_map, "./masks/Both-Fornix_Atlas-Merge.nii.gz")

l_cingulum_map = Nifti1Image(l_cingulum, mamm_body_map.affine) # they have se same affine information
r_cingulum_map = Nifti1Image(r_cingulum, mamm_body_map.affine) # they have se same affine information
nib.save(l_cingulum_map, "./masks/Left-Cingulum_Atlas-Merge.nii.gz")
nib.save(r_cingulum_map, "./masks/Right-Cingulum_Atlas-Merge.nii.gz")