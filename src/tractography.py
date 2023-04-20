import sys
import os
import subprocess
import json
import copy

import elikopy
import elikopy.utils
from dipy.io.streamline import load_tractogram, save_trk
from regis.core import find_transform, apply_transform
from params import get_folder, get_segmentation
from threading import Thread, Semaphore

# absolute_path = os.path.dirname(__file__) # return the abs path of the folder of this file, wherever it is

class ROI:
    def __init__(self, name, path, isCortex) -> None:
        self.name = name
        self.path = path
        self.isCortex = isCortex

    def __str__(self) -> str:
        return self.path
    
frontalLobe = ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis", "parsorbitalis", "parstriangularis", "lateralorbitofrontal", "medialorbitofrontal", "precentral", "paracentral", "frontalpole"]

temporalLobe = ["superiortemporal", "middletemporal", "inferiortemporal", "bankssts", "fusiform", "transversetemporal", "entorhinal", "temporalpole", "parahippocampal"]

occipitalLobe = ["lateraloccipital", "lingual", "cuneus", "pericalcarine"]

tracts = {"stria_terminalis":
            {
                "seed_images": ["amygdala"],
                "include" : [],
                "include_ordered" : ["fornixST", "fornix", "BNST"],
                "exclude" : ["hippocampus", "thalamus", "lateral-ventricle", "Inf-Lat-Vent", "Caudate", "Putamen", "Pallidum", "CSF", "Accumbens-area"]
            },
          "fornix":
            {
                "seed_images": ["hippocampus"],
                "include" : [],
                "include_ordered" : ["fornixST", "fornix", "mammillaryBody"], 
                "exclude" : ["amygdala", "thalamus", "lateral-ventricle", "Inf-Lat-Vent", "Caudate", "Putamen", "Pallidum", "CSF", "Accumbens-area"]
            },
            "thalamus-AntCingCtx":
            {
                "seed_images": ["thalamus"],
                "include" : ["rostralanteriorcingulate", "rostralmiddlefrontal"],
                "include_ordered" : [],
                "exclude" : []
            },
            "thalamus-Insula":
            {
                "seed_images": ["thalamus"],
                "include" : ["insula"],
                "include_ordered" : [],
                "exclude" : []
            },
            # For Association fibers, info taken from Wikipedia and Freesurfer
            "sup-longi-fasci":{ 
                "seed_images" : frontalLobe,
                "include" : occipitalLobe
            },
            "inf-longi-fasci":{ 
                "seed_images" : occipitalLobe,
                "include" : temporalLobe
            },
            "inf-longi-fasci":{ 
                "seed_images" : occipitalLobe,
                "include" : frontalLobe
            },
          }

roi_freesurfer = {
    "hippocampus" : [17, 53],
    "amygdala" : [18, 54],
    "thalamus" : [10, 49],
    "lateral-ventricles" : [4, 43, 5, 44],
    "caudate" : [11, 50],
    "putamen" : [12, 51],
    "pallidum" : [13, 52],
    "csf" : [24],
    "accumbens" : [26, 58],
    "ctx-lh-interval" : [1001, 1035],
    "ctx-rh-interval" : [2001, 2035]
}
roi_num_name = {}

def expand_roi():
    roi_nums_tot = []
    for name, roi_numbers in roi_freesurfer.items():
        if "interval" not in name:
            for roi_num in roi_numbers:
                roi_nums_tot.append(roi_num)
        else:
            roi_nums_tot.extend(list(range(roi_numbers[0], roi_numbers[1]+1)))
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

def freesurfer_mask_extraction(folder_path, seg_path, subj_id):
    for num, name in roi_num_name.items():
        out_path = "%s/subjects/%s/masks/%s_%s_aparc+aseg.nii.gz" % (folder_path, subj_id, subj_id, name)
        cmd = "mri_extract_label -exit_none_found %s/%s/mri/aparc+aseg.mgz %d %s" % (seg_path, subj_id, num, out_path)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        process.wait()
        if process.returncode != 0:
            os.remove(out_path)

def registration(folder_path, subj_id):

    moving_file_FA = folder_path + "/static_files/atlases/FSL_HCP1065_FA_1mm.nii.gz"
    moving_file_T1 = folder_path + "/static_files/atlases/MNI152_T1_05mm.nii.gz"

    static_file_FA = folder_path + "/subjects/" + subj_id + "/dMRI/microstructure/dti/" + subj_id + "_FA.nii.gz"
    static_file_T1 = folder_path + "/T1/" + subj_id + "_T1.nii.gz"

    print(moving_file_FA, moving_file_T1, static_file_FA, static_file_T1)

    if not os.path.isfile(moving_file_FA) or not os.path.isfile(static_file_FA) or not os.path.isfile(moving_file_T1) or not os.path.isfile(static_file_T1):
        print("Images for subject: " + subj_id + " weren't found")
        exit(1)

    mapping_FA = None
    mapping_T1 = None

    for path, dirs, files in os.walk(folder_path + "/static_files/atlases/masks"):
        for file in files: # register all the masks
            if file.endswith(".nii.gz") :

                mask_file = path + "/" + file
                moving_file = mask_file
                output_path = folder_path + "/subjects/" + subj_id + "/masks/" + subj_id + "_" + file

                print("Applying transformation from " + file.split(".")[0])
                if "_FA_" in file:
                    
                    if mapping_FA is None:
                        print("Finding transformation from atlas FA ")
                        mapping_FA = find_transform(moving_file_FA, static_file_FA)
                        print("Transformation found")

                    print("FA:",end=" ")

                    apply_transform(moving_file, mapping_FA, static_file_FA, output_path=output_path, binary=True, binary_thresh=0)
                elif "_T1_" in file:
                    
                    if mapping_T1 is None:
                        print("Finding transformation from atlas T1")
                        mapping_T1 = find_transform(moving_file_T1, static_file_T1)
                        print("Transformation found")
                    
                    print("T1:",end=" ")

                    apply_transform(moving_file, mapping_T1, static_file_T1, output_path=output_path, binary=True, binary_thresh=0)
                print("Transformed")

def get_mask(mask_path):
    roi_names = {}
    roi_names["left"] = {}
    roi_names["right"]= {}

    for path, _, files in os.walk(mask_path):
        for file in files:
            fileName_ext = file.split(".")
            fileName = fileName_ext[0]
            ext = "."+".".join(fileName_ext[1:])
            if ext != ".nii.gz" :
                continue
    
            roiName = fileName.split("_")[1].lower().split("-")
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
                roi_names["left"][name] = ROI(name, path+"/"+file, isCortex)
                roi_names["right"][name] = ROI(name, path+"/"+file, isCortex)
                continue

            roi_names[side][name] = ROI(name, path+"/"+file, isCortex)

    return roi_names

def find_tract(subj_folder_path, subj_id, seed_images, inclusions, inclusions_ordered, exclusions, output_name):
    """
    It's a function that build the bashCommands for the tckgen of mrtrix3 and generate the tracts
    """
    tck_path = subj_folder_path+"/dMRI/tractography/"+output_name+".tck"
    process = None

    bashCommand = ("tckgen -nthreads 8 -algorithm iFOD2 -select 1000 -seeds 500k -max_attempts_per_seed 1000 -angle 42.5 -cutoff 0.12 -seed_unidirectional -stop -fslgrad -force" +
                   subj_folder_path + "/dMRI/preproc/"+subj_id+"_dmri_preproc.bvec " +
                   subj_folder_path + "/dMRI/preproc/"+subj_id+"_dmri_preproc.bval")
    
    for region in seed_images:
        bashCommand += " -seed_image " + region
    for region in inclusions:
        bashCommand += " -include " + region
    for region in inclusions_ordered:
        bashCommand += " -include_ordered " + region
    for region in exclusions:
        bashCommand += " -exclude " + region

    bashCommand += " " + subj_folder_path + "/dMRI/ODF/MSMT-CSD/"+subj_id+"_MSMT-CSD_WM_ODF.nii.gz " + tck_path
    
    print(bashCommand)
    print(" ")

    process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
    process.wait()

    ############ CONVERSION TCK -> TRK #################              
    if not os.path.isfile(tck_path) or process.returncode != 0: # only if there was an error during the tractography, to not block everything
        return
    print("Converting %s" % tck_path)
    tract = load_tractogram(tck_path, subj_folder_path+"/dMRI/preproc/"+subj_id+"_dmri_preproc.nii.gz")
    save_trk(tract, tck_path[:-3]+'trk')

def main():

    ## Getting folder
    folder_path = get_folder(sys.argv)
    
    # check if the user wants to compute the ODF and compute it
    if "-odf" in sys.argv[1:]:
        study = elikopy.core.Elikopy(folder_path, cuda=True, slurm=True, slurm_email="michele.cerra@student.uclouvain.be")

        study.odf_msmtcsd()

    extract_roi = False
    if "-roi" in sys.argv[1:]:
        extract_roi = True  
        seg_path = get_segmentation(sys.argv)
        get_freesurfer_roi_names()

    ## Read the list of subjects and for each subject do the tractography
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as file:
        patient_list = json.load(file)

    # TODO change to patient_list
    for p_code in ["subj00"]:

        subj_folder_path = folder_path + '/subjects/' + p_code
        
        # check if the ODF exist for the subject, otherwise skip subject
        if not os.path.isdir(subj_folder_path + "/dMRI/ODF/MSMT-CSD/") :
            print("multi-tissue orientation distribution function is not found for patient: %s" % (p_code))
            continue

        ############# ROI EXTRACTION ############

        if extract_roi:

            # Do the registration to extract ROI from atlases
            registration(folder_path, p_code)

            # Extract ROI from freesurfer segmentation
            # check if the freesurfer segmentation exist, otherwise skip subject
            # Here we are assuming that the segmentation is already done
            if not os.path.isdir(seg_path + "/" + p_code + "/mri"):
                print("freesurfer segmentation isn't found for paritent: %s" % (p_code))
                continue

            freesurfer_mask_extraction(folder_path, seg_path, p_code)

        roi_names = get_mask(subj_folder_path+"/masks")

        ########### TRACTOGRAPHY ##########
        for zone in tracts.keys():
            for side in ["left", "right"]:
                opts = {}
                opts["seed_images"] = []; opts["include"] = []; opts["include_ordered"] = []; opts["exclude"] = []

                areAllROIs = True

                # convert the option in path of the associated file
                for opt, rois in tracts[zone].items():
                    for roi in rois:
                        # find the file name inside the roi_names
                        if roi.lower() not in roi_names[side]:
                            print("Mask of roi %s isn't found: skipping it" % (roi.lower()))
                            areAllROIs = False
                            continue
                        opts[opt].append(roi_names[side][roi.lower()].path)
                
                if not areAllROIs: # All the mask must be present
                    continue

                if zone != "thalamocortical":
                    # fornix and stria_terminalis case
                    find_tract(copy.deepcopy(subj_folder_path), copy.deepcopy(p_code), copy.deepcopy(opts["seed_images"]), copy.deepcopy(opts["include"]), copy.deepcopy(opts["include_ordered"]), copy.deepcopy(opts["exclude"]), side+"-"+zone)
                else:
                    # thalamus cortical tractography
                    for ctx_roi in roi_names[side].values():
                        if ctx_roi.isCortex:
                            opts["include"].append(ctx_roi.path)

                            find_tract(copy.deepcopy(subj_folder_path), copy.deepcopy(p_code), copy.deepcopy(opts["seed_images"]), copy.deepcopy(opts["include"]), copy.deepcopy(opts["include_ordered"]), copy.deepcopy(opts["exclude"]), side+"-"+zone+"-"+ctx_roi.name)

                            opts["include"].pop() # remove the added ctx seg to analyze the next one

if __name__ == "__main__":
    exit(main())
