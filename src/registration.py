import sys
import os
import json

from regis.core import find_transform, apply_transform
import nibabel as nib
from params import get_arguments


def registration(folder_path):

    ## Read the list of subjects
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    for p_code in ["subj00"]:

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

if __name__ == "__main__":
    ## Getting Parameters
    _, _, _, folder_path = get_arguments(sys.argv)

    exit(registration(folder_path))
