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
        # Non conto lo Stria-Terminalis ma devo scriverlo nella tesi che non l'ho messo.. la ragione e qualche foto
        # Il livello di definizione delle immaggini non permette la tractografia della ST
        # "stria_terminalis":
        #     {
        #         "seed_images": ["amygdala"],
        #         "include_ordered" : ["fornixST", "fornix", "BNST"],
        #         "exclude" : ["hippocampus", "Thalamus-Proper", "Caudate", "Putamen", "Pallidum"]
        #     }

        "fornix":
            {
                "seed_images": ["hippocampus", "amygdala"],
                "include" : ["mammillary-body"],
                "include_ordered" : ["plane-fornix", "plane-ort-fornix", "plane-mammillary-body", "plane1-mammillary-body"], 
                # Change Thalamus-Proper to Thalamus depending on the version of freesurfer
                "exclude" : ["Thalamus-Proper", "Caudate", "Putamen", "Pallidum"],
                "cutoff" : 0.07,
                "angle" : 25
            },

        "thalamus-AntCingCtx":
            {
                "seed_images": ["Thalamus-Proper"],
                "include_ordered" : ["plane-cingulum", "plane-cingulate", "frontal-cingulate"],
                "angle" : 30,
                "cutoff" : 0.07,
            },
        "thalamus-Insula":
            {
                "seed_images": ["Thalamus-Proper"],
                "include" : ["insula"],
                "masks" : ["thalamus-insula-hull-dilated-15"],
                "exclude" : ["hippocampus"],
                "angle" : 20
            },
            
        "sup-longi-fasci":
            { 
                "seed_images" : ["frontal-lobe"],
                "include" : ["parietal-lobe"],
                "masks" : ["cerebral-white-matter", "frontal-lobe", "parietal-lobe"],
                "angle" : 15,
                "cutoff" : 0.09
            },
        "inf-longi-fasci":
            { 
                "seed_images" : ["occipital-lobe"],
                "include" : ["temporal-lobe"],
                "masks" : ["cerebral-white-matter", "occipital-lobe", "temporal-lobe"],
                "angle" : 15,
                "cutoff" : 0.09
            },

        # Non conto l'Inferior front-occipital ma devo scriverlo nella tesi xche non l'ho messo.. la ragione e qualche foto
        # "inf-front-occipital-fasci":
        #     { 
        #         "seed_images" : ["frontal-lobe"],
        #         "include" : ["occipital-lobe"],
        #         "masks" : ["cerebral-white-matter", "frontal-lobe", "occipital-lobe"],
        #         "angle" : 15,
        #         "cutoff" : 0.09
        #     },
          }

# Freesurfer LUT: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
roi_freesurfer = {
    "hippocampus" : [17, 53],
    "amygdala" : [18, 54],
    "thalamus" : [10, 49],
    "caudate" : [11, 50],
    "putamen" : [12, 51],
    "pallidum" : [13, 52],
    "accumbens" : [26, 58],
    "insula" : [1035, 2035],
    "wm" : [2, 41],
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
    "Left-Temporal-Lobe" : [1030, 1015, 1009, 1001, 1007, 1034, 1006, 1033, 1016],
    "Right-Temporal-Lobe" : [2030, 2015, 2009, 2001, 2007, 2034, 2006, 2033, 2016],
    "Left-Parietal-Lobe" : [1008, 1029, 1031, 1025], # Without considering the Post-central
    "Right-Parietal-Lobe" : [2008, 2029, 2031, 2025], # Without considering the Post-central
    "Left-Occipital-Lobe" : [1011, 1013, 1005, 1021],
    "Right-Occipital-Lobe" : [2011, 2013, 2005, 2021],
}

# Change thalamus-proper in thalamus depending on the version of freesurfer
convex_hull = {
    "thalamus-insula-hull" : ["thalamus-proper", "insula"]
}

dilatations = {
    "thalamus-insula-hull" : 15
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

def freesurfer_mask_extraction(folder_path, subj_id):
    ## Registration from T1 to dMRI
    registration_path = folder_path + "/subjects/" + subj_id + "/registration"
    if not os.path.isdir(registration_path):
        os.mkdir(registration_path)

    print("Computing matrix transformation between T1 and dMRI")
    if not os.path.isfile(registration_path + "/transf_dMRI_t1.dat"):
        # Find the transformation matrix
        cmd = "bbregister --s %s --mov %s/subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz --reg %s/transf_dMRI_t1.dat --dti --init-fsl" % (subj_id, folder_path, subj_id, subj_id, registration_path)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        process.wait()
        if process.returncode != 0:
            print("Error freesurfer bbregister")
            return 1
    
    # Apply transformation to T1 to see the result
    print("Apply transformation: T1 to dMRI")
    cmd = "mri_vol2vol --reg %s/transf_dMRI_t1.dat --mov %s/subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz --fstarg --o %s/%s_T1_brain_reg.nii.gz --interp nearest --no-resample --inv" % (registration_path, folder_path, subj_id, subj_id, registration_path, subj_id)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    process.wait()
    if process.returncode != 0:
        print("Error freesurfer mri_vol2vol T1")
        return 1

    # Apply transformation to aseg+aparc
    print("Apply transformation: aparc+aseg to dMRI")
    cmd = "mri_vol2vol --reg %s/transf_dMRI_t1.dat --targ %s/subjects/%s/mri/aparc+aseg.mgz --mov %s/subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz --o %s/aparc+aseg_reg.mgz --interp nearest --no-resample --inv" % (registration_path, folder_path, subj_id, folder_path, subj_id, subj_id, registration_path)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    process.wait()
    if process.returncode != 0:
        print("Error freesurfer mri_vol2vol aseg")
        return 1

    # Normal regions
    for num, name in roi_num_name.items():
        print("Extraction ROI: %s" % name)
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
        out_path = "%s/subjects/%s/masks/%s_%s_aparc+aseg.nii.gz" % (folder_path, subj_id, subj_id, name)
        cmd = "mri_extract_label -exit_none_found %s/aparc+aseg_reg.mgz %s %s" % (registration_path, roi_numbers_string, out_path)
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

def registration(folder_path, subj_id):
    masks_path = folder_path + "/static_files/atlases/masks"
    # Atlas MNI152
    atlas_file = folder_path + "/static_files/atlases/MNI152_T1_1mm_brain.nii.gz"
    atlas_map = ants.image_read(atlas_file)
    # Subject T1 only brain
    subj_t1_file = folder_path + "/subjects/" + subj_id + "/mri/brain.mgz"
    subj_t1_map = ants.image_read(subj_t1_file)
    # Subject dMRI extraction of b0
    subj_dmri_file = folder_path + "/subjects/" + subj_id + "/dMRI/preproc/" + subj_id + "_dmri_preproc.nii.gz"
    subj_b0_file = folder_path + "/subjects/" + subj_id + "/dMRI/preproc/" + subj_id + "_b0_preproc.nii.gz"
    # Here I use nibabel to extract the b0 map
    subj_dmri_map : Nifti1Image = nib.load(subj_dmri_file)
    subj_dmri_numpy = subj_dmri_map.get_fdata() # get the numpy array
    subj_b0_numpy = subj_dmri_numpy[:,:,:,0] # get only the first volume (b0)
    subj_b0_map = Nifti1Image(subj_b0_numpy, subj_dmri_map.affine) # convert to nii.gz image with same affine information of the full dmri map
    nib.save(subj_b0_map, subj_b0_file) # save it 
    subj_b0_map = ants.image_read(subj_b0_file) # load with ants

    # folder to save the registrations
    registration_path = folder_path + "/subjects/" + subj_id + "/registration"
    if not os.path.isdir(registration_path):
        os.mkdir(registration_path)
    if not os.path.isdir(registration_path + "/ants"):
        os.mkdir(registration_path + "/ants")

    cwd = os.getcwd() # save the current working directory
    os.chdir(registration_path + "/ants") # change dir inside /ants

    if not os.path.isfile("./tx_t1_dMRI_1Warp.nii.gz") or not os.path.isfile("./tx_t1_dMRI_0GenericAffine.mat") or not os.path.isfile("./tx_atl_t1_1Warp.nii.gz") or not os.path.isfile("./tx_atl_t1_0GenericAffine.mat"):
    
        # Find transformation

        # Transform Atlas -> T1
        tx_atl_t1 = ants.registration(
            fixed=subj_t1_map, 
            moving=atlas_map, 
            type_of_transform='SyNAggro',
            reg_iterations = [10000, 1000, 100],
            outprefix="tx_atl_t1_"
            )
        # Transform T1 -> dMRI
        tx_t1_dmri = ants.registration(
            fixed=subj_b0_map,
            moving=subj_t1_map,
            type_of_transform="SyNBoldAff",
            reg_iterations = [10000, 1000, 100],
            outprefix="tx_t1_dMRI_"
        )

        # Combine the two transformations
        transform =  tx_t1_dmri["fwdtransforms"] + tx_atl_t1["fwdtransforms"]

    else:
        # Use the transformations already computed
        transform = [
        "tx_t1_dMRI_1Warp.nii.gz",
        "tx_t1_dMRI_0GenericAffine.mat",
        "tx_atl_t1_1Warp.nii.gz",
        "tx_atl_t1_0GenericAffine.mat",
        ]

    # Apply transformation to MNI152 to see the result
    mask_moved = ants.apply_transforms(
        fixed=subj_b0_map,
        moving=atlas_map,
        transformlist=transform,
        interpolator= "nearestNeighbor"
    )
    
    ants.image_write(mask_moved, cwd + "/" + registration_path + "/MNI152_T1_1mm_brain_reg.nii.gz")

    # Apply the transformation to all the mask
    for file in os.listdir(cwd + "/" + masks_path):
        ext = ".".join(file.split(".")[-2:])
        mask_file = cwd + "/" + masks_path + "/" + file
        if os.path.isfile(mask_file) and ext == "nii.gz":

            mask_map = ants.image_read(mask_file)

            print("Applying transformation: " + file.split(".")[0])
            mask_moved = ants.apply_transforms(
                fixed=subj_b0_map,
                moving=mask_map,
                transformlist=transform,
                interpolator= "nearestNeighbor"
            )
            ants.image_write(mask_moved, cwd + "/" + folder_path + "/subjects/" + subj_id + "/masks/" + subj_id + "_" + file)
    
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

def find_tract(subj_folder_path, subj_id, seed_images, inclusions, inclusions_ordered, exclusions, masks, angle, cutoff, stop, act, output_name):
    """
    It's a function that build the bashCommands for the tckgen of mrtrix3 and generate the tracts
    """
    tck_path = subj_folder_path+"/dMRI/tractography/"+output_name+".tck"
    process = None

    bashCommand = "tckgen -nthreads 4 -algorithm iFOD2 -seeds 10M -max_attempts_per_seed 1000 -seed_unidirectional -force"

    if stop:
        bashCommand += " -stop"
    if act:
        bashCommand += " -act " + subj_folder_path + "/dMRI/5tt/subj00_5tt.nii.gz"

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
        q1 = np.quantile(lengths, 0.20)
        q3 = np.quantile(lengths, 0.80)
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
    

def compute_tracts(p_code, folder_path, extract_roi, tract, onlySide:str):
    print("Working on %s" % p_code)

    subj_folder_path = folder_path + '/subjects/' + p_code
    seg_path = folder_path + "/subjects"
        
    # check if the ODF exist for the subject, otherwise skip subject
    if not os.path.isdir(subj_folder_path + "/dMRI/ODF/MSMT-CSD/") :
        print("multi-tissue orientation distribution function is not found for patient: %s" % (p_code))
        return 1

    if not os.path.isdir(subj_folder_path + "/dMRI/tractography/"):
        os.mkdir(subj_folder_path + "/dMRI/tractography/")
    if not os.path.isdir(subj_folder_path + "/masks/"):
        os.mkdir(subj_folder_path + "/masks/")

    ############# ROI EXTRACTION ############
    if extract_roi:
        # extract ROI from atlases
        print("MNI152 roi extraction on %s" % p_code)
        if registration(folder_path, p_code) is not None:
            return 1

        # Extract ROI from freesurfer segmentation
        # check if the freesurfer segmentation exist, otherwise skip subject
        # Here we are assuming that the segmentation is already done
        if not os.path.isdir(seg_path + "/" + p_code + "/mri"):
            print("freesurfer segmentation isn't found for patient: %s" % (p_code))
            return 1
        
        os.environ["SUBJECTS_DIR"] = seg_path

        get_freesurfer_roi_names()

        print("Freesurfer roi extraction on %s" % p_code)
        if freesurfer_mask_extraction(folder_path, p_code) is not None:
            print("Error freesurfer extraction or registration")
            return 1

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
            opts["angle"] = 45
            opts["cutoff"] = 0.1
            opts["stop"] = True
            opts["act"] = False

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
                elif type(rois) is int or type(rois) is float or type(rois) is bool:
                    opts[opt] = rois
            
            if not areAllROIs: # All the mask must be present
                continue

            output_name = side+"-"+zone
            output_path = subj_folder_path+"/dMRI/tractography/"+output_name+".tck"

            print(json.dumps(opts, indent=2))

            # forward try at maximum 3 times to find something
            for _ in range(3):
                output_path_forward = find_tract(subj_folder_path, p_code, opts["seed_images"], opts["include"], opts["include_ordered"], opts["exclude"], opts["masks"], opts["angle"], opts["cutoff"], opts["stop"], opts["act"], output_name+"_to")

                trk = load_tractogram(output_path_forward, subj_folder_path + "/dMRI/ODF/MSMT-CSD/"+p_code+"_MSMT-CSD_WM_ODF.nii.gz")
                nTracts = get_streamline_count(trk)
                if nTracts > 0:
                    break

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

            # backward try at maximum 3 times to find something
            for _ in range(3):
                output_path_backward = ""
                if len(opts["include_ordered"]) > 0 and len(opts["seed_images"]) > 1:
                    output_path_backward = subj_folder_path+"/dMRI/tractography/"+output_name+"_from"
                    cmd = "tckedit -force" # command to do the union of all the tracts
        
                    # The reverse of more seed regions
                    for i, seed_path in enumerate(opts["seed_images"]):
                        optsReverse["include_ordered"].append(seed_path)
                        cmd = cmd + " " + output_path_backward + str(i) + ".tck"
                        # # #   

                        find_tract(subj_folder_path, p_code, optsReverse["seed_images"], optsReverse["include"], optsReverse["include_ordered"], opts["exclude"], opts["masks"], opts["angle"], opts["cutoff"], opts["stop"], opts["act"], output_name+"_from" + str(i))

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
                    output_path_backward = find_tract(subj_folder_path, p_code, optsReverse["seed_images"], optsReverse["include"], optsReverse["include_ordered"], opts["exclude"], opts["masks"], opts["angle"], opts["cutoff"], opts["stop"], opts["act"], output_name+"_from")

                # check that there is at least some tract, for 3 times
                trk = load_tractogram(output_path_backward, subj_folder_path + "/dMRI/ODF/MSMT-CSD/"+p_code+"_MSMT-CSD_WM_ODF.nii.gz")
                nTracts = get_streamline_count(trk)
                if nTracts > 0:
                    break

            # select both tracks 
            if os.path.isfile(output_path_forward) and os.path.isfile(output_path_backward):
                removeOutliers(output_path_forward)
                removeOutliers(output_path_backward)

                # Union of the removed tracts in forward and backward
                cmd = "tckedit -force %s %s %s" % (output_path_forward[:-4] + "_rmvd.tck", output_path_backward[:-4] + "_rmvd.tck", output_path[:-4] + "_rmvd.tck")
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                process.wait()

                os.remove(output_path_forward[:-4] + "_rmvd.tck"); 
                os.remove(output_path_backward[:-4] + "_rmvd.tck")

                convertTck2Trk(subj_folder_path, p_code, output_path[:-4] + "_rmvd.tck")

                # Union of the track in forward and backward
                cmd = "tckedit -force %s %s %s" % (output_path_forward, output_path_backward, output_path)
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                process.wait()
                os.remove(output_path_forward); os.remove(output_path_backward)

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

    extract_roi = False
    if "-roi" in sys.argv[1:]:
        extract_roi = True  

    tract = False
    if "-tract" in sys.argv[1:]:
        tract = True

    side = ""
    if "-side" in sys.argv[1:]:
        parIdx = sys.argv.index("-side") + 1 # the index of the parameter after the option
        side = sys.argv[parIdx]

    compute_tracts(p, folder_path, extract_roi, tract, side)

if __name__ == "__main__":
    exit(main())
