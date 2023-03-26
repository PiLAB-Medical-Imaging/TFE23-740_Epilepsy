import sys
import os
import json

from regis.core import find_transform, apply_transform
import nibabel as nib
from params import get_arguments

structures = ["fornix", "stria_terminalis", "thalamocortical"]

def main():
    ## Getting Parameters
    _, _, _, folder_path = get_arguments(sys.argv)

    ## Read the list of subjects
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    for p_code in ["subj00"]:

        moving_file = folder_path + "/static_files/atlases/FSL_HCP1065_FA_1mm.nii.gz"
        static_file = folder_path + "/subjects/" + p_code + "/dMRI/microstructure/dti/" + p_code + "_FA.nii.gz"

        if not os.path.isfile(moving_file) or not os.path.isfile(static_file):
            print("Atlas of FA image for subject: " + p_code + " wasn't found")
            continue

        print("Finding transformation from atlas to " + p_code)

        mapping = find_transform(moving_file, static_file)

        print("Transformation found")

        for _, dirs, _ in os.walk(folder_path + "/static_files/atlases"): # a trick to get all the sub-folders

            for struct in dirs: # for each folder in the atlases
                for file in os.listdir( folder_path + "/static_files/atlases/" + struct ): # register all the masks
                    if file.endswith(".nii.gz") :

                        mask_file = folder_path + "/static_files/atlases/" + struct + "/" + file
                        moving_file = mask_file
                        output_path = folder_path + "/subjects/" + p_code + "/masks/" + p_code + "_template_" + file

                        print("Applying transformation from " + struct + " " + file.split(".")[0], end="")
                        apply_transform(moving_file, mapping, static_file, output_path=output_path, binary=True, binary_thresh=0)
                        print("Transformed")

            break # Otherwise the first for will explore all the sub directories, but we have already done

if __name__ == "__main__":
    exit(main())