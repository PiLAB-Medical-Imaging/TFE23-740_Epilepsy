#%% IMPORTS %%#
import ants
import os
from tqdm.auto import tqdm

#%% Get the paths %%#
study_dir = os.path.join(os.environ.get("HOME"), "Dropbox (Politecnico Di Torino Studenti)", "thesis", "code", "Epilepsy-dMRI-VNS", "study")
subjs_dir = os.path.join(study_dir, "subjects")
fixed_dir = os.path.join(study_dir, "static_files", "atlases", "FSL_HCP1065_FA_1mm.nii.gz")

image_paths = {}

for i in range(23+1):
    subj_id = "VNSLC_%02.f" % i
    subj_dir = os.path.join(subjs_dir, subj_id)

    if not os.path.isdir(subj_dir):
        continue
    
    image_paths[subj_id] = {
        "FA": f"{subj_dir}/dMRI/microstructure/dti/{subj_id}_FA.nii.gz",
        "MD": f"{subj_dir}/dMRI/microstructure/dti/{subj_id}_MD.nii.gz",
        "AD": f"{subj_dir}/dMRI/microstructure/dti/{subj_id}_AD.nii.gz",
        "RD": f"{subj_dir}/dMRI/microstructure/dti/{subj_id}_RD.nii.gz",
        "wFA": f"{subj_dir}/dMRI/microstructure/diamond/{subj_id}_diamond_wFA.nii.gz",
        "wMD": f"{subj_dir}/dMRI/microstructure/diamond/{subj_id}_diamond_wMD.nii.gz",
        "wAD": f"{subj_dir}/dMRI/microstructure/diamond/{subj_id}_diamond_wAD.nii.gz",
        "wRD": f"{subj_dir}/dMRI/microstructure/diamond/{subj_id}_diamond_wRD.nii.gz",
        "diamond_frac_csf": f"{subj_dir}/dMRI/microstructure/diamond/{subj_id}_diamond_frac_csf.nii.gz",
        "icvf": f"{subj_dir}/dMRI/microstructure/noddi/{subj_id}_noddi_icvf.nii.gz",
        "odi": f"{subj_dir}/dMRI/microstructure/noddi/{subj_id}_noddi_odi.nii.gz",
        "fextra": f"{subj_dir}/dMRI/microstructure/noddi/{subj_id}_noddi_fextra.nii.gz",
        "fiso": f"{subj_dir}/dMRI/microstructure/noddi/{subj_id}_noddi_fiso.nii.gz",
        "wfvf": f"{subj_dir}/dMRI/microstructure/mf/{subj_id}_mf_wfvf.nii.gz",
        "fvf_tot": f"{subj_dir}/dMRI/microstructure/mf/{subj_id}_mf_fvf_tot.nii.gz", 
        "mf_frac_csf": f"{subj_dir}/dMRI/microstructure/mf/{subj_id}_mf_frac_csf.nii.gz",        
    }

    for path in image_paths[subj_id].values():
        assert os.path.exists(path)
            

#%% Test one image %%#

subj_id = "VNSLC_01"
type_of_transform = 'DenseRigid'

output_dir = os.path.join(subjs_dir, subj_id, "registration", "FAinMNI_rigid")
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

FA_vol = ants.image_read(image_paths[subj_id]["diamond_frac_csf"])
# FA_fixed = ants.image_read(fixed_dir)
# FA_fixed.plot(overlay=FA_vol)

# mytx = ants.registration(
#     fixed=FA_fixed,
#     moving=FA_vol,
#     type_of_transform=type_of_transform,
#     outprefix=output_dir+"/"
# )
# print(mytx)
# warped_moving = mytx['warpedmovout']
# fwdtransforms = mytx["fwdtransforms"]

# FA_fixed.plot(overlay=warped_moving)

# transformed = ants.apply_transforms(
#     fixed=FA_fixed,
#     moving=FA_vol,
#     transformlist=fwdtransforms,
#     interpolator="linear",
# )
# 
# ants.image_write(transformed, os.path.join(subjs_dir, subj_id, "registration", "FAinMNI_rigid", image_paths[subj_id]["diamond_frac_csf"].split("/")[-1]))


#%% transform images %% 

type_of_transform = 'DenseRigid'

for subj_id in tqdm(image_paths.keys()):
    output_dir = os.path.join(subjs_dir, subj_id, "registration", "FAinMNI_rigid")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    FA_vol = ants.image_read(image_paths[subj_id]["FA"])
    FA_fixed = ants.image_read(fixed_dir)

    mytx = ants.registration(
        fixed=FA_fixed,
        moving=FA_vol,
        type_of_transform=type_of_transform,
        outprefix=output_dir+"/"
    )

    fwdtransforms = mytx["fwdtransforms"]

    for image_path in image_paths[subj_id].values():
        moving_volume = ants.image_read(image_path)

        transformed = ants.apply_transforms(
            fixed=FA_fixed,
            moving=moving_volume,
            transformlist=fwdtransforms,
            interpolator="gaussian",
        )

        ants.image_write(transformed, os.path.join(subjs_dir, subj_id, "registration", "FAinMNI_rigid", image_path.split("/")[-1]))

