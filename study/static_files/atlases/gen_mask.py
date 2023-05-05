import numpy as np
import nibabel as nib

from nibabel import Nifti1Image

# Prob masks atlas Juelich
mamm_body_map = nib.load("./Both-Mammillary-body_Juelich.nii.gz")
mamm_body = mamm_body_map.get_fdata()
fornix_juelich = nib.load("./Both-Fornix_Juelich.nii.gz").get_fdata()

# Prob masks atlas Xtract
l_fornix_xtract = nib.load("./Left-Fornix_Xtract.nii.gz").get_fdata()
r_fornix_xtract = nib.load("./Right-Fornix_Xtract.nii.gz").get_fdata()

# Non prob atlas Jhu
fornix_jhu = nib.load("./Both-Fornix_Jhu-labels.nii.gz").get_fdata()
l_fornixSt_jhu = nib.load("./Left-FornixST_Jhu-labels.nii.gz").get_fdata()
r_fornixSt_jhu = nib.load("./Right-FornixST_Jhu-labels.nii.gz").get_fdata()

# Non prob atlas Talairach
l_mamm_body_tal = nib.load("./Left-Mammillary-body_Talairach.nii.gz").get_fdata()
r_mamm_body_tal = nib.load("./Right-Mammillary-body_Talairach.nii.gz").get_fdata()

# Threshold the prob masks
prob_masks = [mamm_body, fornix_juelich, l_fornix_xtract, r_fornix_xtract]
for mask in prob_masks:
    mask[mask < 15] = 0

# Merging of fornix masks
fornix_masks = [fornix_juelich, l_fornix_xtract, r_fornix_xtract, fornix_jhu, l_fornixSt_jhu, r_fornixSt_jhu]
fornix = np.zeros(fornix_juelich.shape, dtype=np.float64) # empty mask
for mask in fornix_masks:
    fornix += mask

# Merging of mammillary body
mamm_masks = [mamm_body, l_mamm_body_tal, r_mamm_body_tal]
mamm = np.zeros(mamm_body.shape, dtype=np.float64) # empty mask
for mask in mamm_masks:
    mamm += mask

# Saving 
mamm_mask_map = Nifti1Image(mamm, mamm_body_map.affine)
nib.save(mamm_mask_map, "./masks/Both-Mammillary-body_Atlas-Merge.nii.gz")

fornix_mask_map = Nifti1Image(fornix, mamm_body_map.affine) # they have se same affine informations
nib.save(fornix_mask_map, "./masks/Both-Fornix_Atlas-Merge.nii.gz")
