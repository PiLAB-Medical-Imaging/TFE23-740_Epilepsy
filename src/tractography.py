import sys
import os
import subprocess
import json
import elikopy
import elikopy.utils
import ants
import nibabel as nib

from dipy.io.streamline import load_tractogram, save_trk
from dipy.tracking.utils import length
from unravel.utils import *
from nibabel import Nifti1Image
from params import *
from skimage.morphology import convex_hull_image

# absolute_path = os.path.dirname(__file__) # return the abs path of the folder of this file, wherever it is

class ROI:
    def __init__(self, name, path, isCortex) -> None:
        self.name = name
        self.path = path
        self.isCortex = isCortex

    def __str__(self) -> str:
        return self.path

tracts = {
        "antThalRadiation": 
            {
                "seed_images": ["thalamus"],
                "include_ordered" : ["AntLimbIntCapsule", "frontal-lobe"],
                "stop" : False,
                "act" : True,
                "select" : "10k"
            },
        "postThalRadiation-parital": 
            {
                "seed_images": ["thalamus"],
                "include_ordered" : ["PostLimbIntCapsule", "parietal-lobe"],
                "exclude" : ["VLa", "VLp"], # they connect to the SLF, it's a false tract
                "stop" : False,
                "act" : True,
                "angle" : 10,
                "select" : "10k"
            },
        "postThalRadiation-occipital": 
            {
                "seed_images": ["thalamus"],
                "include_ordered" : ["PostLimbIntCapsule", "occipital-lobe"],
                "exclude" : ["VLa", "VLp", "plane1-SLF1"], # they connect to the SLF, it's a false tract
                "stop" : False,
                "act" : True,
                "angle" : 10,
                "select" : "10k"
            },
        "supThalRadiation": 
            {
                "seed_images": ["thalamus"],
                "include_ordered" : ["PostLimbIntCapsule", "gyrus-central"],
                "stop" : False,
                "act" : True,
                "select" : "10k"
            },
        "infThalRadiation-insula": 
            {
                "seed_images": ["thalamus"],
                "include_ordered" : ["RetroLenticularIntCapsule", "insula"],
                "exclude" : ["MGN", "temporal-lobe-dilated-1", "parietal-lobe-dilated-1", "gyrus-central-dilated-1", "frontal-lobe-dilated-1", "supramarginal-dilated-1"],
                "stop" : False,
                "act" : True,
                "select" : "10k"
            },
        "infThalRadiation-temporal": 
            {
                "seed_images": ["thalamus"],
                "include_ordered" : ["RetroLenticularIntCapsule", "temporal-lobe"],
                "stop" : False,
                "act" : True,
                "select" : "10k"
            },

        # This Tract doesn't exist, it connects to the cingulate, that not start from the thalamus    
        # "thalamus-AntCingCtx":
        #     {
        #         "seed_images": ["Thalamus"],
        #         # "include_ordered" : ["plane-cingulum", "plane-cingulate", "frontal-cingulate"],
        #         "include" : ["frontal-cingulate"],
        #         "stop" : False, 
        #         "act" : True,
        #         "select" : "10k"
        #     },

        # DEPRECATED
        # "thalamus-Insula":
        #     {
        #         "seed_images": ["Thalamus"],
        #         "include" : ["insula"],
        #         "masks" : ["thalamus-insula-hull-dilated-15"],
        #         "stop" : False,
        #         "act" : True,
        #         "select" : "10k"
        #     },
            
        "sup-longi-fasci-1":
            { 
                "seed_images" : ["frontal-lobe"],
                "include_ordered" : ["plane-SLF1", "parietal-lobe"],
                "masks" : ["cerebral-white-matter", "frontal-lobe", "parietal-lobe"],
                "angle" : 10,
                "stop" : False,
                "act" : True, 
                "select" : "10k"
            },
        "sup-longi-fasci-2":
            { 
                "seed_images" : ["frontal-lobe"],
                "include_ordered" : ["plane-SLF2", "parietal-lobe"],
                "masks" : ["cerebral-white-matter", "frontal-lobe", "parietal-lobe"],
                "angle" : 10,
                "stop" : False,
                "act" : True, 
                "select" : "10k"
            },
        "sup-longi-fasci-3":
            { 
                "seed_images" : ["frontal-lobe"],
                "include_ordered" : ["plane-SLF3", "parietal-lobe"],
                "masks" : ["cerebral-white-matter", "frontal-lobe", "parietal-lobe"],
                "angle" : 10,
                "stop" : False,
                "act" : True, 
                "select" : "10k"
            },
        "inf-longi-fasci":
            { 
                "seed_images" : ["occipital-lobe"],
                "include" : ["temporal-lobe"],
                "masks" : ["cerebral-white-matter", "occipital-lobe", "temporal-lobe"],
                "angle" : 10,
                "stop" : False,
                "act" : True,
                "select" : "10k"
            },
        
        "fornix":
            {
                "seed_images" : ["plane-mammillary-body"],
                "include_ordered" : ["plane-ort-fornix", "plane-fornix", "hippocampus"],
                "exclude" : ["Thalamus-eroded-1", "Lateral-Ventricle-eroded-1"],
                "stop" : True,
                "select" : "10k"
            },
          }

# Freesurfer LUT: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
roi_freesurfer = {
    "hippocampus" : [17, 53],
    "amygdala" : [18, 54],
    "lateral-ventricle" : [4, 43],
    "insula" : [1035, 2035],
    "wm" : [2, 41],
    "VLa" : [8128, 8228],
    "VLp" : [8129, 8229],
    "MGN" : [8115, 8215],
}
roi_num_name = {}

union_reg = {
    "Left-Frontal-Cingulate" : [1026, 1002],
    "Right-Frontal-Cingulate" : [2026, 2002],
    # Lobe information taken from:
    # Freesurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation
    # Wikipedia: https://en.wikipedia.org/wiki/Association_fiber
    "Left-Frontal-Lobe" : [1028, 1027, 1003, 1018, 1019, 1020, 1012, 1014, 1032], # Without considering the Pre-central and Para-central
    "Right-Frontal-Lobe" : [2028, 2027, 2003, 2018, 2019, 2020, 2012, 2014, 2032], # Without considering the Pre-central and Para-central
    "Left-Temporal-Lobe" : [1030, 1015, 1009, 1001, 1007, 1034, 1006, 1033], # Without considering parahippocampal
    "Right-Temporal-Lobe" : [2030, 2015, 2009, 2001, 2007, 2034, 2006, 2033], # Without considering parahippocampal
    "Left-Parietal-Lobe" : [1008, 1029, 1025], # Without considering the Post-central and Supramarginal
    "Right-Parietal-Lobe" : [2008, 2029, 2025], # Without considering the Post-central and Supramarginal
    "Left-Occipital-Lobe" : [1011, 1013, 1005, 1021],
    "Right-Occipital-Lobe" : [2011, 2013, 2005, 2021],
    "Left-Supramarginal" : [1031],
    "Right-Supramarginal" : [2031],
    # Other for thalamic radiations
    # Thalamocortical connections: https://www.ncbi.nlm.nih.gov/books/NBK546699/#
    "Left-Gyrus-Central" : [1022, 1024], # Without considering the Para-Central
    "Right-Gyrus-Central" : [2022, 2024], # Without considering the Para-Central
    # Thalamus nuclei
    # Full thalamus 
    "Left-Thalamus" : [8103, 8104, 8105, 8106, 8108, 8109, 8110, 8111, 8112, 8113, 8115, 8116, 8117, 8118, 8119, 8120, 8121, 8122, 8123, 8125, 8126, 8127, 8128, 8129, 8130, 8133, 8134],
    "Right-Thalamus" : [8203, 8204, 8205, 8206, 8208, 8209, 8210, 8211, 8212, 8213, 8215, 8216, 8217, 8218, 8219, 8220, 8221, 8222, 8223, 8225, 8226, 8227, 8228, 8229, 8230, 8233, 8234],
    # Anterior 
    # "Left-Anterior-Thalamus" : [8103, 8112, 8113, 8116],
    # "Right-Anterior-Thalamus" : [8203, 8212, 8213, 8216],
    # Ventral lateral
    # "Left-Ventral-Lateral" : [8128, 8129, 8133],
    # "Right-Ventral-Lateral" : [8228, 8229, 8233],
    # Ventral nuclei
    # "Left-Ventral-Nuclei" : [8126, 8127, 8128, 8129, 8130, 8133],
    # "Right-Ventral-Nuclei" : [8226, 8227, 8228, 8229, 8230, 8233],
    # Posterior nuclei
    # "Left-Posterior-Nuclei" : [8109, 8111, 8115, 8120, 8121, 8122, 8123],
    # "Right-Posterior-Nuclei" : [8209, 8211, 8215, 8220, 8221, 8222, 8223]
}   

# Change thalamus-proper in thalamus depending on the version of freesurfer
convex_hull = {
    # "thalamus-insula-hull" : ["thalamus", "insula"],
    # "insula-putamen-hull" : ["insula", "putamen"]
}

sottractions = {
    #"insula-putamen-hull-in" : ["insula-putamen-hull", "insula", "putamen"],
}

dilatations = {
    #"thalamus-insula-hull" : 15
    "temporal-lobe" : 1,
    "parietal-lobe" : 1,
    "gyrus-central" : 1,
    "frontal-lobe" : 1,
    "supramarginal" : 1
}

erosions = {
    "Lateral-Ventricle" : 1,
    "Thalamus" : 1
}

def expand_roi():
    roi_nums_tot = []
    for name, roi_numbers in roi_freesurfer.items():
        if "lobe" in name.lower() or "cingulate" in name.lower():
            continue 
        for roi_num in roi_numbers:
            roi_nums_tot.append(roi_num)
    return roi_nums_tot

def get_freesurfer_roi_names():
    colorLUT = os.getenv('FREESURFER_HOME') + "/FreeSurferColorLUT.txt"
    roi_nums_tot = expand_roi()
    roi_nums_tot.sort()
    k = 0
    
    with open(colorLUT, "r") as f:
        for line in f.readlines():
            elems = line.split()
            if len(elems) == 0:
                continue
            if elems[0] == "#" or not elems[0].isdigit():
                continue
            roi_num = int(elems[0])
            roi_name = elems[1]
            if roi_num == roi_nums_tot[k]:
                roi_num_name[roi_num] = roi_name
                k += 1
            if k == len(roi_nums_tot):
                break

def save_convex_hull(mask_paths, out_path):
    if len(mask_paths) == 0:
        return
    
    affine_info = None
    union_mask = None # Here we do the sum of the mask and then we transform them in binary
    for i, mask_path in enumerate(mask_paths):
        mask_map : Nifti1Image = nib.load(mask_path)
        mask_np = mask_map.get_fdata()

        if i == 0: # get the affine of the first element, we suppose that all the mask are in the same space, with same affine information
            affine_info = mask_map.affine
            union_mask = mask_np
        else:
            union_mask = union_mask + mask_np
    
    union_mask[union_mask > 0] = 1 # binaryze it 
    hull = convex_hull_image(union_mask)

    hull = hull + 0 # tranform in integer
    hull = hull.astype("float64") # tranform in float becayse Nifti1Image wants floats
    hull_map = Nifti1Image(hull, affine_info)
    nib.save(hull_map, out_path)

def save_sottraction(orig_path : str, sub_paths : list, out_path : str):
    # the affine information MUST be equal for all the maps
    assert len(sub_paths) > 0

    # load the origin mask
    origin_map : Nifti1Image = nib.load(orig_path)
    origin_np = origin_map.get_fdata()
    origin_np[origin_np > 1] = 1 # Binarization

    # load the sottraent masks and do the sottraction
    for sub_path in sub_paths:
        sub_map : Nifti1Image = nib.load(sub_path)
        sub_np = sub_map.get_fdata()
        sub_np[sub_np > 1] = 1 # Binarization
        # sottraction
        origin_np = origin_np - sub_np

    origin_np[origin_np > 1] = 1 # Binarization of the output
    origin_np = origin_np.astype("float64")

    origin_map = Nifti1Image(origin_np, origin_map.affine)
    nib.save(origin_map, out_path)

def freesurfer_mask_extraction(folder_path, subj_id):
    registration_path = folder_path + "/subjects/" + subj_id + "/registration"

    # Normal regions
    for num, name in roi_num_name.items():
        print("Extraction ROI: %s" % name)
        if num > 8000:
            out_path = "%s/subjects/%s/masks/%s_%s_ThalamicNuclei.nii.gz" % (folder_path, subj_id, subj_id, name)

            cmd = "mri_extract_label -exit_none_found %s/ThalamicNuclei_reg.nii.gz %d %s" % (registration_path, num, out_path)
        else:
            out_path = "%s/subjects/%s/masks/%s_%s_aparc+aseg.nii.gz" % (folder_path, subj_id, subj_id, name)

            cmd = "mri_extract_label -exit_none_found %s/aparc+aseg_reg.mgz %d %s" % (registration_path, num, out_path)

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        process.wait()
        if process.returncode != 0:
            os.remove(out_path)

    # Union Regions
    for name, roi_numbers in union_reg.items():
        print("Union ROI: %s" % name)
        roi_numbers_string = " ".join(str(i) for i in roi_numbers)
        seg = -1
        for num in roi_numbers:
            if seg == -1:
                seg = 2 if num >= 8000 else 1
                continue
            if (seg == 1 and num >= 8000) or (seg == 2 and num < 8000):
                print("The numeber of regions inserted are from different segmentation file")
                return 1
        if seg == 1:
            out_path = "%s/subjects/%s/masks/%s_%s_aparc+aseg.nii.gz" % (folder_path, subj_id, subj_id, name)
            cmd = "mri_extract_label -exit_none_found %s/aparc+aseg_reg.mgz %s %s" % (registration_path, roi_numbers_string, out_path)
        elif seg == 2:
            out_path = "%s/subjects/%s/masks/%s_%s_ThalamicNuclei.nii.gz" % (folder_path, subj_id, subj_id, name)
            cmd = "mri_extract_label -exit_none_found %s/ThalamicNuclei_reg.nii.gz %s %s" % (registration_path, roi_numbers_string, out_path)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        process.wait()
        if process.returncode != 0:
            os.remove(out_path)

    roi_names = get_mask(folder_path+"/subjects/"+ subj_id +"/masks", subj_id)

    # Convex hull regions
    for roi_hull, rois in convex_hull.items():
        print("Convex Hull ROI: %s" % roi_hull)
        for side in ["left", "right"]:
            roi_paths = []
            for roi in rois:
                if roi in roi_names[side].keys():
                    roi_paths.append(roi_names[side][roi].path)
                else:
                    print("Error convex hull: %s ; %s doesn't found" % (roi_hull, roi))
                    return 1
            save_convex_hull(roi_paths, "%s/subjects/%s/masks/%s_%s-%s_aparc+aseg.nii.gz" % (folder_path, subj_id, subj_id, side, roi_hull))

    roi_names = get_mask(folder_path+"/subjects/"+ subj_id +"/masks", subj_id)

    # Dilated regions
    masks_path = "%s/subjects/%s/masks" % (folder_path, subj_id)
    file_extracted_names = os.listdir(masks_path)
    for file_name in file_extracted_names:
        file_path = masks_path + "/" + file_name
        ext = "."+".".join(file_name.split(".")[-2:])
        if os.path.isfile(file_path) and ext == ".nii.gz":
            roi_name = "-".join(file_name.split(subj_id + "_")[1].split("_")[0].split("-")[1:]).lower()

            for roi_dila in dilatations.keys():
                if roi_dila.lower() == roi_name.lower():
                    print("Dilatation ROI: %s" % roi_dila)
                    dilatation_cicle = dilatations[roi_dila]
                    dilated_roi_name = file_name.split(subj_id + "_")[1].split("_")[0] + "-dilated-" + str(dilatation_cicle)
                    output_path = "%s/subjects/%s/masks/%s_%s_aparc+aseg.nii.gz" % (folder_path, subj_id, subj_id, dilated_roi_name)

                    cmd = "maskfilter -force -npass %d %s dilate %s" % (dilatation_cicle, file_path, output_path)
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                    process.wait()
                    break

    # Erosion regions
    masks_path = "%s/subjects/%s/masks" % (folder_path, subj_id)
    file_extracted_names = os.listdir(masks_path)
    for file_name in file_extracted_names:
        file_path = masks_path + "/" + file_name
        ext = "."+".".join(file_name.split(".")[-2:])
        if os.path.isfile(file_path) and ext == ".nii.gz":
            roi_name = "-".join(file_name.split(subj_id + "_")[1].split("_")[0].split("-")[1:]).lower()

            for roi_ero in erosions.keys():
                if roi_ero.lower() == roi_name.lower():
                    print("Erosion ROI: %s" % roi_ero)
                    erosion_cicle = erosions[roi_ero]
                    eroded_roi_name = file_name.split(subj_id + "_")[1].split("_")[0] + "-eroded-" + str(erosion_cicle)
                    output_path = "%s/subjects/%s/masks/%s_%s_aparc+aseg.nii.gz" % (folder_path, subj_id, subj_id, eroded_roi_name)

                    cmd = "maskfilter -force -npass %d %s erode %s" % (erosion_cicle, file_path, output_path)
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                    process.wait()
                    break

    roi_names = get_mask(folder_path+"/subjects/"+ subj_id +"/masks", subj_id)
    
    # Sottracted regions
    for roi_sottr, rois in sottractions.items():
        print("Sottraction ROI", roi_sottr)
        for side in ["left", "right"]:
            roi_paths = []
            for roi in rois:
                if roi in roi_names[side].keys():
                    roi_paths.append(roi_names[side][roi].path)
                else:
                    print("Error convex hull: %s ; %s doesn't found" % (roi_sottr, roi))
                    return 1
            save_sottraction(roi_paths[0], roi_paths[1:], "%s/subjects/%s/masks/%s_%s-%s_aparc+aseg.nii.gz" % (folder_path, subj_id, subj_id, side, roi_sottr))

def registration(folder_path, subj_id, force):
    masks_path = folder_path + "/static_files/atlases/masks"
    dmri_path = folder_path + "/subjects/" + subj_id + "/dMRI/preproc"
    # Atlas MNI152
    atlas_file = folder_path + "/static_files/atlases/MNI152_T1_1mm_brain.nii.gz"
    atlas_map = ants.image_read(atlas_file)
    # Subject dMRI extraction of b0
    subj_dmri_file = dmri_path + "/" + subj_id + "_dmri_preproc.nii.gz"
    subj_b0_file = dmri_path + "/" + subj_id + "_b0_preproc.nii.gz"
    # Here I use MRtrix3 to extract the mean of the b0 maps
    cmd = "dwiextract %s -fslgrad %s/%s_dmri_preproc.bvec %s/%s_dmri_preproc.bval - -bzero | mrmath - mean %s -axis 3 -force" % (subj_dmri_file, dmri_path, subj_id, dmri_path, subj_id, subj_b0_file)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    process.wait()
    if process.returncode != 0:
        print("Error mean of b0")
        return 1

    ## Registration from T1 to dMRI
    registration_path = folder_path + "/subjects/" + subj_id + "/registration"
    if not os.path.isdir(registration_path):
        os.mkdir(registration_path)
    if not os.path.isdir(registration_path + "/ants"):
        os.mkdir(registration_path + "/ants")

    print("Computing matrix transformation between T1 and dMRI")
    if force or not os.path.isfile(registration_path + "/transf_dMRI_t1.dat"):
        # Find the transformation matrix
        cmd = "bbregister --s %s --mov %s --reg %s/transf_dMRI_t1.dat --dti --init-fsl" % (subj_id, subj_b0_file, registration_path)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        process.wait()
        if process.returncode != 0:
            print("Error freesurfer bbregister")
            return 1
    
    # Apply transformation to T1
    subj_t1_map_reg = registration_path + "/" + subj_id + "_T1_brain_reg.nii.gz"
    print("Apply transformation: T1 to dMRI")
    cmd = "mri_vol2vol --reg %s/transf_dMRI_t1.dat --targ %s/freesurfer/%s/mri/brain.mgz --mov %s --o %s --interp nearest --no-resample --inv" % (registration_path, folder_path, subj_id, subj_b0_file, subj_t1_map_reg)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    process.wait()
    if process.returncode != 0:
        print("Error freesurfer mri_vol2vol T1")
        return 1
    subj_t1_map_reg = ants.image_read(subj_t1_map_reg) # load with ants

    # Apply transformation to aseg+aparc
    print("Apply transformation: aparc+aseg to dMRI")
    cmd = "mri_vol2vol --reg %s/transf_dMRI_t1.dat --targ %s/freesurfer/%s/mri/aparc+aseg.mgz --mov %s --o %s/aparc+aseg_reg.mgz --interp nearest --no-resample --inv" % (registration_path, folder_path, subj_id, subj_b0_file, registration_path)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    process.wait()
    if process.returncode != 0:
        print("Error freesurfer mri_vol2vol aseg")
        return 1

    # Apply transformation to 5tt
    print("Apply transformation: 5tt to dMRI")
    cmd = "mri_vol2vol --reg %s/transf_dMRI_t1.dat --targ %s/subjects/%s/5tt/%s_5tt.nii.gz --mov %s --o %s/5tt_reg.nii.gz --interp nearest --no-resample --inv" % (registration_path, folder_path, subj_id, subj_id, subj_b0_file, registration_path)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    process.wait()
    if process.returncode != 0:
        print("Error freesurfer mri_vol2vol 5tt")
        return 1
    
    # Apply transformation to Thalamus segmentation
    print("Apply transformation: ThalamicNuclei to dMRI")
    cmd = "mri_vol2vol --reg %s/transf_dMRI_t1.dat --targ %s/freesurfer/%s/mri/ThalamicNuclei.v12.T1.FSvoxelSpace.mgz --mov %s --o %s/ThalamicNuclei_reg.nii.gz --interp nearest --no-resample --inv" % (registration_path, folder_path, subj_id, subj_b0_file, registration_path)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    process.wait()
    if process.returncode != 0:
        print("Error freesurfer mri_vol2vol 5tt")
        return 1
    
    cwd = os.getcwd() # save the current working directory
    os.chdir(registration_path + "/ants") # change dir inside /ants

    print("Computing matrix transformation between MNI and T1_reg")
    # if not os.path.isfile("./tx_t1_dMRI_1Warp.nii.gz") or not os.path.isfile("./tx_t1_dMRI_0GenericAffine.mat") or not os.path.isfile("./tx_atl_t1_1Warp.nii.gz") or not os.path.isfile("./tx_atl_t1_0GenericAffine.mat"):
    if force or not os.path.isfile("./tx_atl_t1_1Warp.nii.gz") or not os.path.isfile("./tx_atl_t1_0GenericAffine.mat"):
    
        # Find transformation

        # Transform Atlas -> T1 reg
        tx_atl_t1 = ants.registration(
            fixed=subj_t1_map_reg, 
            moving=atlas_map, 
            type_of_transform='ElasticSyN',
            outprefix="tx_atl_t1_"
            )
        # Transform T1 -> dMRI
        #tx_t1_dmri = ants.registration(
        #    fixed=subj_b0_map,
        #    moving=subj_t1_map,
        #    type_of_transform="SyNBoldAff",
        #    reg_iterations = [10000, 1000, 100],
        #    outprefix="tx_t1_dMRI_"
        #)

        # Combine the two transformations
        #transform =  tx_t1_dmri["fwdtransforms"] + tx_atl_t1["fwdtransforms"]
        transform = tx_atl_t1["fwdtransforms"]

    else:
        # Use the transformations already computed
        transform = [
        #"tx_t1_dMRI_1Warp.nii.gz",
        #"tx_t1_dMRI_0GenericAffine.mat",
        "tx_atl_t1_1Warp.nii.gz",
        "tx_atl_t1_0GenericAffine.mat",
        ]

    # Apply transformation to MNI152 to see the result
    print("Apply transformation: MNI to T1_reg")
    mask_moved = ants.apply_transforms(
        fixed=subj_t1_map_reg,
        moving=atlas_map,
        transformlist=transform,
        interpolator="gaussian"
    )
    
    ants.image_write(mask_moved, cwd + "/" + registration_path + "/MNI152_T1_1mm_brain_reg.nii.gz")

    # Apply the transformation to all the mask
    for file in os.listdir(cwd + "/" + masks_path):
        ext = ".".join(file.split(".")[-2:])
        mask_file = cwd + "/" + masks_path + "/" + file
        if os.path.isfile(mask_file) and ext == "nii.gz":

            # Use ANT to do the registration MNI -> T1

            mask_map = ants.image_read(mask_file)

            print("Applying transformation: " + file.split(".")[0])
            mask_moved = ants.apply_transforms(
                fixed=subj_t1_map_reg,
                moving=mask_map,
                transformlist=transform,
                interpolator= "nearestNeighbor"
            )
            ants.image_write(mask_moved, cwd + "/" + folder_path + "/subjects/" + subj_id + "/masks/" + subj_id + "_" + file)

            # # use the freesurfer transformation for the registration T1 -> dMRI
            # os.chdir(cwd)
# 
            # cmd = "mri_vol2vol --reg %s/transf_dMRI_t1.dat --targ %s/%s_%s --mov %s/subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz --o %s/subjects/%s/masks/%s_%s --interp nearest --no-resample --inv" % (registration_path, registration_path, subj_id, file, folder_path, subj_id, subj_id, folder_path, subj_id, subj_id, file)
            # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            # process.wait()
            # if process.returncode != 0:
            #     print("Error freesurfer mri_vol2vol aseg")
            #     return 1
            # 
            # os.chdir(registration_path + "/ants")
    
    os.chdir(cwd) # return to the cwd

def get_mask(mask_path, subj_id):
    roi_names = {}
    roi_names["left"] = {}
    roi_names["right"]= {}

    for file in os.listdir(mask_path):
        if not file.endswith(".nii.gz"):
            continue
        fileName = file.split(".")[0]

        roiName = fileName.split(subj_id+"_")[1].split("_")[0].lower().split("-")
        if "left" not in roiName and "right" not in roiName and "lh" not in roiName and "rh" not in roiName and "both" not in roiName:
            if roiName[0] == "csf" or len(roiName) >= 2 :
                roiName.insert(0, "both")
            else :
                continue
        if len(roiName) < 2:
                continue
        name = None; side = None; isCortex = None

        if "ctx" != roiName[0]:
            side, name = roiName[0], "-".join(roiName[1:])
            isCortex = False
        else :
            side, name = roiName[1], "-".join(roiName[2:])
            side = "left" if side == "lh" else "right"
            isCortex = True
        
        if side == "both":
            roi_names["left"][name] = ROI(name, mask_path+"/"+file, isCortex)
            roi_names["right"][name] = ROI(name, mask_path+"/"+file, isCortex)
            continue

        roi_names[side][name] = ROI(name, mask_path+"/"+file, isCortex)

    return roi_names

def find_tract(subj_folder_path, subj_id, seeds:str, seed_images, select:str, inclusions, inclusions_ordered, exclusions, masks, angle, cutoff, stop, act, output_name):
    """
    It's a function that build the bashCommands for the tckgen of mrtrix3 and generate the tracts
    """
    tck_path = subj_folder_path+"/dMRI/tractography/"+output_name+".tck"
    process = None

    bashCommand = "tckgen -nthreads 4 -algorithm iFOD2 -seeds %s -select %s -max_attempts_per_seed 1000 -seed_unidirectional -force" % (seeds, select)

    if stop:
        bashCommand += " -stop"
    if act:
        bashCommand += " -act " + subj_folder_path + "/registration/5tt_reg.nii.gz -backtrack -crop_at_gmwmi"

    bashCommand += " -angle " + str(angle)
    bashCommand += " -cutoff " + str(cutoff)
    bashCommand += (" -fslgrad " + subj_folder_path + "/dMRI/preproc/"+subj_id+"_dmri_preproc.bvec " +
                                   subj_folder_path + "/dMRI/preproc/"+subj_id+"_dmri_preproc.bval")
    
    for region in seed_images:
        bashCommand += " -seed_image " + region
    for region in inclusions:
        bashCommand += " -include " + region
    for region in inclusions_ordered:
        bashCommand += " -include_ordered " + region
    for region in exclusions:
        bashCommand += " -exclude " + region
    for region in masks:
        bashCommand += " -mask " + region

    bashCommand += " " + subj_folder_path + "/dMRI/ODF/MSMT-CSD/"+subj_id+"_MSMT-CSD_WM_ODF.nii.gz " + tck_path
    
    print(bashCommand)
    print(" ")

    process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
    process.wait()

    return tck_path

def convertTck2Trk(subj_folder_path, subj_id, tck_path):
    if not os.path.isfile(tck_path): # only if there was an error during the tractography, to not block everything
        return
    print("Converting %s" % tck_path)
    tract = load_tractogram(tck_path, subj_folder_path+"/dMRI/preproc/"+subj_id+"_dmri_preproc.nii.gz")
    save_trk(tract, tck_path[:-3]+'trk')

def removeOutliers(tck_path):
    """
    Remotion of outliers streamlines with the IQR rule
    """
    if not os.path.isfile(tck_path): # only if there was an error during the tractography, to not block everything
        return
    
    trk_path_noExt = tck_path[:-4]
    removed_path = trk_path_noExt+"_rmvd.tck"


    bundle  = nib.streamlines.load(tck_path).streamlines
    lengths = list(length(bundle))
    if len(lengths) > 0:
        q1 = np.quantile(lengths, 0.30)
        q3 = np.quantile(lengths, 0.70)
    else:
        q1 = 0
        q3 = 0
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr

    if upper < 0: upper = 0
    if lower < 0: lower = 0
    print(lower, upper)

    # Save the difference tracts (The ones that have been removed)
    cmd = "tckedit -maxlength %f -force %s %s_lower.tck && tckedit -minlength %f -force %s %s_upper.tck && tckedit -force %s_lower.tck %s_upper.tck %s" % (lower, tck_path, trk_path_noExt, upper, tck_path, trk_path_noExt, trk_path_noExt, trk_path_noExt, removed_path)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    process.wait()
    os.remove(trk_path_noExt+"_lower.tck"); os.remove(trk_path_noExt+"_upper.tck")

    # Remove the outliers
    cmd = "tckedit -minlength %f -maxlength %f -force %s %s" % (lower, upper, tck_path, tck_path)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    process.wait()
    
def compute_tracts(p_code, folder_path, compute_5tt, extract_roi, tract, force, onlySide:str):
    print("Working on %s" % p_code)

    subj_folder_path = folder_path + '/subjects/' + p_code
    freesurfer_subj_folder_path = folder_path + '/freesurfer/' + p_code
    freesurfer_path = folder_path + "/freesurfer"
        
    # check if the ODF exist for the subject, otherwise skip subject
    if not os.path.isdir(subj_folder_path + "/dMRI/ODF/MSMT-CSD/") :
        print("multi-tissue orientation distribution function is not found for patient: %s" % (p_code))
        raise Exception

    if not os.path.isdir(subj_folder_path + "/dMRI/tractography/"):
        os.mkdir(subj_folder_path + "/dMRI/tractography/")
    if not os.path.isdir(subj_folder_path + "/masks/"):
        os.mkdir(subj_folder_path + "/masks/")
    if not os.path.isdir(subj_folder_path + "/5tt/"):
        os.mkdir(subj_folder_path + "/5tt/")

    ############# 5TT COMPUTATION ##########
    if compute_5tt:
        if force or not os.path.isfile("%s/5tt/%s_5tt.nii.gz" % (subj_folder_path, p_code)):
            cmd = "5ttgen fsl %s/mri/orig.mgz %s/5tt/%s_5tt.nii.gz -t2 %s/mri/T2.mgz -nocrop -force && mrconvert %s/5tt/%s_5tt.nii.gz %s/5tt/%s_5tt.nii.gz -strides %s/mri/brain.mgz -force" % (freesurfer_subj_folder_path, subj_folder_path, p_code, freesurfer_subj_folder_path, subj_folder_path, p_code, subj_folder_path, p_code, freesurfer_subj_folder_path)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            process.wait()
            if process.returncode != 0:
                print("Error 5tt computation")
                raise Exception

    ############# ROI EXTRACTION ############
    if extract_roi:
        os.environ["SUBJECTS_DIR"] = freesurfer_path

        # extract ROI from atlases
        print("MNI152 roi extraction on %s" % p_code)
        if registration(folder_path, p_code, force) is not None:
            raise Exception

        # Extract ROI from freesurfer segmentation
        # check if the freesurfer segmentation exist, otherwise skip subject
        # Here we are assuming that the segmentation is already done
        if not os.path.isdir(freesurfer_path + "/" + p_code + "/mri"):
            print("freesurfer segmentation isn't found for patient: %s" % (p_code))
            raise Exception

        get_freesurfer_roi_names()

        print("Freesurfer roi extraction on %s" % p_code)
        if freesurfer_mask_extraction(folder_path, p_code) is not None:
            print("Error freesurfer extraction or registration")
            raise Exception

    roi_names = get_mask(subj_folder_path+"/masks", p_code)

    ########### TRACTOGRAPHY ##########
    if not tract:
        return

    for zone in tracts.keys():
        for side in ["left", "right"]:
            if onlySide!="" and onlySide.lower() != side.lower():
                continue

            opts = {}

            opts["seed_images"] = []
            opts["include"] = []
            opts["include_ordered"] = []
            opts["exclude"] = []
            opts["masks"] = []
            opts["angle"] = 15
            opts["cutoff"] = 0.1
            opts["stop"] = True
            opts["act"] = not opts["stop"]
            opts["select"] = "10k"

            areAllROIs = True

            # convert the option in path of the associated file
            for opt, rois in tracts[zone].items():
                if type(rois) is list:
                    for roi in rois:
                        # find the file name inside the roi_names
                        if roi.lower() not in roi_names[side]:
                            print("Mask of roi %s isn't found: skipping %s" % (roi.lower(), zone))
                            areAllROIs = False
                            continue
                        opts[opt].append(roi_names[side][roi.lower()].path)
                elif type(rois) is int or type(rois) is float or type(rois) is bool or type(rois) is str:
                    opts[opt] = rois
            
            if not areAllROIs: # All the mask must be present
                continue

            output_name = side+"-"+zone
            output_path = subj_folder_path+"/dMRI/tractography/"+output_name+".tck"

            print(json.dumps(opts, indent=2))

            # decrement the cutoff to find a solution with more noise
            while opts["cutoff"] >= 0.01:
                output_path_cutoff = find_tract(subj_folder_path, p_code, "500k", opts["seed_images"], "1", opts["include"], opts["include_ordered"], opts["exclude"], opts["masks"], opts["angle"], opts["cutoff"], opts["stop"], opts["act"], output_name+"_findCut")
                trk = load_tractogram(output_path_cutoff, subj_folder_path + "/dMRI/ODF/MSMT-CSD/"+p_code+"_MSMT-CSD_WM_ODF.nii.gz")
                nTracts = get_streamline_count(trk)
                if nTracts > 0:
                    break
                opts["cutoff"] -= 0.01
            os.remove(output_path_cutoff)

            if opts["cutoff"] < 0.01:
                continue

            output_path_forward = find_tract(subj_folder_path, p_code, "10M", opts["seed_images"], opts["select"], opts["include"], opts["include_ordered"], opts["exclude"], opts["masks"], opts["angle"], opts["cutoff"], opts["stop"], opts["act"], output_name+"_to")
            
            optsReverse = {}
            if len(opts["include_ordered"]) == 0: 
                optsReverse["seed_images"] = opts["include"]
                optsReverse["include"] = opts["seed_images"]
                optsReverse["include_ordered"] = []
            elif len(opts["seed_images"]) == 1:
                optsReverse["seed_images"] = []
                optsReverse["seed_images"].append(opts["include_ordered"][-1])
                optsReverse["include_ordered"] = opts["include_ordered"][::-1][1:]
                optsReverse["include_ordered"].extend(opts["seed_images"])
                optsReverse["include"] = opts["include"]
            else:
                optsReverse["seed_images"] = []
                optsReverse["seed_images"].append(opts["include_ordered"][-1])
                optsReverse["include_ordered"] = opts["include_ordered"][::-1][1:]
                optsReverse["include"] = opts["include"]

            # decrement the cutoff to find a solution with more noise
            output_path_backward = ""
            if len(opts["include"]) + len(opts["include_ordered"]) > 0 : # at leat one region must be included otherwise is a single verse tractography
                if len(opts["include_ordered"]) > 0 and len(opts["seed_images"]) > 1:
                    output_path_backward = subj_folder_path+"/dMRI/tractography/"+output_name+"_from"
                    cmd = "tckedit -force" # command to do the union of all the tracts

                    # The reverse of more seed regions
                    for i, seed_path in enumerate(opts["seed_images"]):
                        optsReverse["include_ordered"].append(seed_path)
                        cmd = cmd + " " + output_path_backward + str(i) + ".tck"
                        # # #   

                        find_tract(subj_folder_path, p_code, "10M", optsReverse["seed_images"], opts["select"], optsReverse["include"], optsReverse["include_ordered"], opts["exclude"], opts["masks"], opts["angle"], opts["cutoff"], opts["stop"], opts["act"], output_name+"_from" + str(i))

                        # # #
                        optsReverse["include_ordered"].pop()

                    # Union of the tracts
                    cmd = cmd + " " + output_path_backward + ".tck"
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                    process.wait()

                    # remove the temp file
                    for i, seed_path in enumerate(opts["seed_images"]):
                        os.remove(output_path_backward + str(i) + ".tck")

                    output_path_backward = output_path_backward + ".tck"

                else:
                    # The reverse of one seed region
                    output_path_backward = find_tract(subj_folder_path, p_code, "10M", optsReverse["seed_images"], opts["select"], optsReverse["include"], optsReverse["include_ordered"], opts["exclude"], opts["masks"], opts["angle"], opts["cutoff"], opts["stop"], opts["act"], output_name+"_from")

            # select both tracks 
            if os.path.isfile(output_path_forward) and os.path.isfile(output_path_backward):
                removeOutliers(output_path_forward)
                removeOutliers(output_path_backward)

                # Union of the removed tracts in forward and backward
                cmd = "tckedit -force %s %s %s" % (output_path_forward[:-4] + "_rmvd.tck", output_path_backward[:-4] + "_rmvd.tck", output_path[:-4] + "_rmvd.tck")
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                process.wait()

                os.remove(output_path_forward[:-4] + "_rmvd.tck")
                os.remove(output_path_backward[:-4] + "_rmvd.tck")

                convertTck2Trk(subj_folder_path, p_code, output_path[:-4] + "_rmvd.tck")

                # Union of the track in forward and backward
                cmd = "tckedit -force %s %s %s" % (output_path_forward, output_path_backward, output_path)
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                process.wait()
                os.remove(output_path_forward); os.remove(output_path_backward)

                convertTck2Trk(subj_folder_path, p_code, output_path)
            elif os.path.isfile(output_path_forward):
                removeOutliers(output_path_forward)

                cmd = "tckedit -force %s %s" % (output_path_forward[:-4] + "_rmvd.tck", output_path[:-4] + "_rmvd.tck")
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                process.wait()

                os.remove(output_path_forward[:-4] + "_rmvd.tck")

                convertTck2Trk(subj_folder_path, p_code, output_path[:-4] + "_rmvd.tck")

                cmd = "tckedit -force %s %s" % (output_path_forward, output_path)
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                process.wait()
                os.remove(output_path_forward)

                convertTck2Trk(subj_folder_path, p_code, output_path)

def main():
    
    ## Getting folder
    folder_path = get_folder(sys.argv)

    ## Get the patient
    p = get_patient(sys.argv)
    
    # check if the user wants to compute the ODF and compute it
    if "-odf" in sys.argv[1:]:
        study = elikopy.core.Elikopy(folder_path, cuda=True, slurm=True, slurm_email="michele.cerra@student.uclouvain.be")

        study.odf_msmtcsd()

    compute_5tt = False
    if "-5tt" in sys.argv[1:]:
        compute_5tt = True

    extract_roi = False
    if "-roi" in sys.argv[1:]:
        extract_roi = True  

    tract = False
    if "-tract" in sys.argv[1:]:
        tract = True

    force = False
    if "-force" in sys.argv[1:]:
        force = True

    side = ""
    if "-side" in sys.argv[1:]:
        parIdx = sys.argv.index("-side") + 1 # the index of the parameter after the option
        side = sys.argv[parIdx]

    compute_tracts(p, folder_path, compute_5tt, extract_roi, tract, force, side)

if __name__ == "__main__":
    exit(main())
