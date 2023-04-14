import sys
import os
import subprocess
import json

import elikopy
import elikopy.utils
from dipy.io.streamline import load_tractogram, save_trk
from regis.core import find_transform, apply_transform
from params import get_arguments

absolute_path = os.path.dirname(__file__) # return the abs path of the folder of this file, wherever it is

masks = ["fornix", "cortex", "stria_terminalis"]

tracts = ["Left-Fornix",
          "Right-Fornix",
          "Left-ST",
          "Right-ST",
          "Left-Thalamocortical", 
          "Right-Thalamocortical"]

tracts_roi = {
    "fornix" : "17 53",
    "stria_terminalis" : "18 54",
    "cortex" : "10 49",
    "cortex_interval1" : "1001 1035",
    "cortex_interval2" : "2001 2035"
}

# TODO put the future freesurfer_roi_extraction inside this file

def registration(folder_path, subj_id):

    ## Read the list of subjects
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f) 
    ## For now the code is for just a patient at the time
    for p_code in [subj_id]:

        moving_file_FA = folder_path + "/static_files/atlases/FSL_HCP1065_FA_1mm.nii.gz"
        moving_file_T1 = folder_path + "/static_files/atlases/MNI152_T1_0.5mm.nii.gz"

        static_file_FA = folder_path + "/subjects/" + p_code + "/dMRI/microstructure/dti/" + p_code + "_FA.nii.gz"
        static_file_T1 = folder_path + "/T1/" + p_code + "_T1.nii.gz"

        if not os.path.isfile(moving_file_FA) or not os.path.isfile(static_file_FA) or not os.path.isfile(moving_file_T1) or not os.path.isfile(static_file_T1):
            print("Images for subject: " + p_code + " weren't found")
            continue

        print("Finding transformation from atlas to " + p_code)

        mapping_FA = find_transform(moving_file_FA, static_file_FA)
        mapping_T1 = find_transform(moving_file_T1, static_file_T1)

        print("Transformation found")
    
        for path, dirs, files in os.walk(folder_path + "/static_files/atlases"): # for each folder in the atlases
            dir_name = path.split("/")[-1]

            curr_path = path.split("/atlases")[1]
            
            # if the sub dirs don't exist, create them inside the subject folder
            for sub_dir in dirs:
                sub_dir_path = folder_path + "/subjects/" + p_code + "/masks" + curr_path + "/" + sub_dir
                if not os.path.isdir(sub_dir_path):
                    os.mkdir(sub_dir_path)

            if len(dirs) == 0:
                for file in files: # register all the masks
                    if file.endswith(".nii.gz") :

                        mask_file = path + "/" + file
                        moving_file = mask_file
                        output_path = folder_path + "/subjects/" + p_code + "/masks" + curr_path + "/" + p_code + "_" + file

                        print("Applying transformation from " + file.split(".")[0], end=" ")
                        if dir_name == "FA":
                            print("FA:",end=" ")
                            apply_transform(moving_file, mapping_FA, static_file_FA, output_path=output_path, binary=True, binary_thresh=0)
                        elif dir_name == "T1":
                            print("T1:",end=" ")
                            apply_transform(moving_file, mapping_T1, static_file_T1, output_path=output_path, binary=True, binary_thresh=0)
                        print("Transformed")

def main():
    ## Getting Parameters
    onSlurm, slurmEmail, cuda, folder_path = get_arguments(sys.argv)

    ## Init
    study = elikopy.core.Elikopy(folder_path, cuda=cuda, slurm=onSlurm, slurm_email=slurmEmail)

    # check if the user wants to compute the ODF
    if "-odf" in sys.argv[1:]:
        study.odf_msmtcsd()

    # TODO write the code to get the seg_path from the arguments
    seg_path = "./seg_subjs"

    ## Read the list of subjects and for each subject do the tractography
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as file:
        patient_list = json.load(file)

    # TODO this kind of operation can be done parallely for each subject
    # TODO change to patient_list
    for p_code in ["subj00"]:

        subj_folder_path = folder_path + '/subjects/' + p_code
        
        # check if the ODF exist for the subject, otherwise skip subject
        if not os.path.isdir(subj_folder_path + "/dMRI/ODF/MSMT-CSD/") :
            print("multi-tissue orientation distribution function is not found for patient: %s" % (p_code))
            continue

        # check if the freesurfer segmentation exist, otherwise skip subject
        if not os.path.isdir(seg_path + "/" + p_code + "/mri"):
            print("freesurfer segmentation isn't found for paritent: %s" % (p_code))

        # make the directory if they don't exist
        for mask in masks:
            mask_path = subj_folder_path + "/masks/" + mask
            if not os.path.isdir(mask_path):
                os.mkdir(mask_path)

        # TODO even this part of registration and ROI extraction can be done parallelly
        # Do the registration to extract ROI from atlases
        registration(folder_path, p_code)
        # Extract ROI from freesurfer segmentation
        for roi_name, vals in tracts_roi.items():
            if "cortex_interval" not in roi_name:
                bashCommand = absolute_path + "/freesurfer_roi_extraction.sh" + " -s " + p_code + " -o " + subj_folder_path + "/mask/" + roi_name + " " + vals
            else:
                roi_name = roi_name.split("_")[0]
                bashCommand = absolute_path + "/freesurfer_roi_extraction.sh" + " -s " + p_code + " -o " + subj_folder_path + "/mask/" + roi_name + " -i " + vals
            
            process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=subprocess.STDOUT, stderr=subprocess.STDOUT)

        #  Oly after did the registration and the extraction
        for tract in tracts:
            # TODO non funziona.. bisogna prima trovare i tratti che vogliamo studiar
            bashCommand= "tckgen -nthreads 4 -algorithm iFOD2 -select 10 -seeds 1M -seed_image left_thalamus.nii.gz -seed_unidirectional -include ACC.nii.gz -stop -fslgrad subj00_dmri_preproc.bvec subj00_dmri_preproc.bval subj00_MSMT-CSD_WM_ODF.nii.gz trcActFsl.tck"

        # TODO these processes should be run inside the CECI cluster through the batch system
        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=subprocess.STDOUT, stderr=subprocess.STDOUT)

        # conversion from .tck to .trk

        tract = load_tractogram(tck_path, dwi_path)

        save_trk(tract, tck_path[:-3]+'trk')

if __name__ == "__main__":
    exit(main())
