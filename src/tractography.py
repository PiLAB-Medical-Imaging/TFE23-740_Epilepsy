import sys
import os
import subprocess
import json

import elikopy
import elikopy.utils
from dipy.io.streamline import load_tractogram, save_trk
from regis.core import find_transform, apply_transform
from params import get_arguments, get_segmentation

absolute_path = os.path.dirname(__file__) # return the abs path of the folder of this file, wherever it is

class ROI:
    def __init__(self, name, path, isCortex) -> None:
        self.name = name
        self.path = path
        self.isCortex = isCortex

    def __str__(self) -> str:
        return self.path

tracts = {"fornix":
            {
                "seed_images": ["amygdala"],
                "include" : [],
                "include_ordered" : ["fornixST", "BNST"],
                "exclude" : ["hippocampus", "fornix"]
            },
          "stria_terminalis":
            {
                "seed_images": ["hippocampus"],
                "include" : [],
                "include_ordered" : ["fornixST", "fornix"],
                "exclude" : ["amygdala", "BNST"]
            },
          # "thalamocortical": {
          #       "seed_images": ["thalamus"],
          #       # here there is a different method.. try all the cortex segments
          #   }
          }

tracts_roi = {
    "fornix" : "17 53",
    "stria_terminalis" : "18 54",
    "cortex" : "10 49",
    "cortex-interval1" : "1001 1035",
    "cortex-interval2" : "2001 2035"
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

        mapping_FA = None
        mapping_T1 = None
    
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
                        if "_FA_" in file:
                            print("FA:",end=" ")
                            
                            if mapping_FA is None:
                                print("Finding transformation from atlas FA to " + p_code)
                                mapping_FA = find_transform(moving_file_FA, static_file_FA)
                                print("Transformation found")

                            apply_transform(moving_file, mapping_FA, static_file_FA, output_path=output_path, binary=True, binary_thresh=0)
                        elif "_T1_" in file:
                            print("T1:",end=" ")

                            if mapping_T1 is None:
                                print("Finding transformation from atlas T1 to " + p_code)
                                mapping_T1 = find_transform(moving_file_T1, static_file_T1)
                                print("Transformation found")

                            apply_transform(moving_file, mapping_T1, static_file_T1, output_path=output_path, binary=True, binary_thresh=0)
                        print("Transformed")

def get_roi_names(mask_path):
    roi_names = {}
    roi_names["left"] = {}
    roi_names["right"]= {}

    for path, _, files in os.walk(mask_path):
        dir_name = path.split("/")[-1]
        if dir_name in tracts_roi:
            for file in files:
                no_ext = file.split(".")[0]

                roi_side_name = no_ext.split("_")[-1].lower()
                name = None; side = None; isCortex = None

                if "ctx" not in roi_side_name:
                    side, name = roi_side_name.split("-")
                    isCortex = False
                else :
                    _, side, name = roi_side_name.split("-")
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

    bashCommand = ("tckgen -nthreads 4 -algorithm iFOD2 -select 1000 -seeds 1M -seed_unidirectional -stop -fslgrad " +
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

    process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True)

    return tck_path, process

def main():
    ## Getting Parameters
    onSlurm, slurmEmail, cuda, folder_path = get_arguments(sys.argv)

    ## Init
    study = elikopy.core.Elikopy(folder_path, cuda=cuda, slurm=onSlurm, slurm_email=slurmEmail)

    seg_path = get_segmentation(sys.argv)
    
    extract_roi = False

    # check if the user wants to compute the ODF and compute it
    if "-odf" in sys.argv[1:]:
        study.odf_msmtcsd()

    if "-roi" in sys.argv[1:]:
        extract_roi = True

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
        # Here we are assuming that the segmentation is already done
        # TODO we can add an option to ask if we want to compute the segmentation in this code
        if not os.path.isdir(seg_path + "/" + p_code + "/mri"):
            print("freesurfer segmentation isn't found for paritent: %s" % (p_code))
            continue

        ############# ROI EXTRACTION ############

        if extract_roi:
            # make the directory if they don't exist
            for mask in tracts_roi.keys():
                mask_path = subj_folder_path + "/masks/" + mask.split("-")[0]
                if not os.path.isdir(mask_path):
                    os.mkdir(mask_path)

            # TODO even this part of registration and ROI extraction can be done parallelly 
            # Do the registration to extract ROI from atlases
            registration(folder_path, p_code)
            # Extract ROI from freesurfer segmentation
            for roi_name, vals in tracts_roi.items():
                if "cortex-interval" not in roi_name:
                    # use absolute_path if it doesn't work
                    bashCommand = "./src/freesurfer_roi_extraction.sh" + " -s " + p_code + " -o " + subj_folder_path + "/masks/" + roi_name + " " + vals
                else:
                    roi_name = roi_name.split("-")[0]
                    bashCommand = "./src/freesurfer_roi_extraction.sh" + " -s " + p_code + " -o " + subj_folder_path + "/masks/" + roi_name + " -i " + vals

                print(bashCommand)
                process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True)
                output, error = process.communicate() # this is necessary to not run everything in parallel

        roi_names = get_roi_names(subj_folder_path+"/masks")

        ########### TRACTOGRAPHY ##########
        tck_path_processes = {} # to run all the tractographies in parallel
        for zone in tracts.keys():
            for side in ["left", "right"]:
                opts = {}
                opts["seed_images"] = []; opts["include"] = []; opts["include_ordered"] = []; opts["exclude"] = []

                # convert the option in path of the associated file
                for opt, rois in tracts[zone].items():
                    for roi in rois:
                        # find the file name inside the roi_names
                        opts[opt].append(roi_names[side][roi.lower()].path)

                tck_path = ""; process = None

                if zone != "thalamocortical":
                    # fornix and stria_terminalis case
                    tck_path, process = find_tract(subj_folder_path, p_code, opts["seed_images"], opts["include"], opts["include_ordered"], opts["exclude"], side+"-"+zone)
                    
                    # add to the list of processes
                    tck_path_processes[tck_path] = process
                else:
                    # thalamus cortical tractography
                    for ctx_roi in roi_names[side].values():
                        if ctx_roi.isCortex:
                            opts["include"].append(ctx_roi.path)

                            tck_path, process = find_tract(subj_folder_path, p_code, opts["seed_images"], opts["include"], opts["include_ordered"], opts["exclude"], side+"-"+zone+"-"+ctx_roi.name)

                            # add to the list of processes
                            tck_path_processes[tck_path] = process

                            opts["include"].pop() # remove the added ctx seg to analyze the next one

        ############ CONVERSION TCK -> TRK #################
        for tck_path, process  in tracts:           
            # wait the end of the process
            process.communicate()

            tract = load_tractogram(tck_path, subj_folder_path+"/dMRI/preproc/"+p_code+"_dmri_preproc.nii.gz")
            save_trk(tract, tck_path[:-3]+'trk')

if __name__ == "__main__":
    exit(main())
