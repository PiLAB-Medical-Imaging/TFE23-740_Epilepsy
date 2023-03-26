import sys
import os
import subprocess
import json

import elikopy
import elikopy.utils
from dipy.io.streamline import load_tractogram, save_trk
from params import get_arguments

tracts = ["Left-Fornix", "Right-Fornix", "Left-ST", "Right-ST", "Left-Thamalocortical", "Right-Thalamocortial"]

def main():
    ## Getting Parameters
    computeODF = False
    onSlurm, slurmEmail, cuda, folder_path = get_arguments(sys.argv)

    ## Check if the used wants to compute the ODF before
    if "-odf" in sys.argv[1:]:
        computeODF = True

    ## Init
    study = elikopy.core.Elikopy(folder_path, cuda=cuda, slurm=onSlurm, slurm_email=slurmEmail)

    # check if the user wants to compute the ODF
    if computeODF:
        study.odf_msmtcsd()

    ## Read the list of subjects and for each subject do the tractography
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as f:
        patient_list = json.load(f)

    for p_code in patient_list:
        
        # check if the ODF exist for the subject, otherwise skip subject
        if not os.path.isdir(folder_path + '/subjects/' + p_code + "/dMRI/ODF/MSMT-CSD/") :
            print("multi-tissue orientation distribution function didn't found")
            continue

        for tract in tracts:
            # TODO non funziona.. bisogna prima trovare i tratti che vogliamo studiar
            bashCommand= "tckgen -nthreads 4 -algorithm iFOD2 -select 10 -seeds 1M -seed_image left_thalamus.nii.gz -seed_unidirectional -include ACC.nii.gz -stop -fslgrad subj00_dmri_preproc.bvec subj00_dmri_preproc.bval subj00_MSMT-CSD_WM_ODF.nii.gz trcActFsl.tck"

        process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True, stdout=tracking_log, stderr=subprocess.STDOUT)

        # conversion from .tck to .trk

        tract = load_tractogram(tck_path, dwi_path)

        save_trk(tract, tck_path[:-3]+'trk')

if __name__ == "__main__":
    exit(main())
