import sys
import os
import elikopy
import json

from params import get_folder, get_segmentation
from elikopy.utils import submit_job, get_job_state

absolute_path = os.path.dirname(__file__) # return the abs path of the folder of this file, wherever it is


def main():

    ## Getting folder
    folder_path = get_folder(sys.argv)
    
    # check if the user wants to compute the ODF and compute it
    if "-odf" in sys.argv[1:]:
        study = elikopy.core.Elikopy(folder_path, cuda=True, slurm=True, slurm_email="michele.cerra@student.uclouvain.be")

        study.odf_msmtcsd()

    time = [0, 1]

    extract_roi = False
    seg_path = ""
    if "-roi" in sys.argv[1:]:
        extract_roi = True  
        seg_path = get_segmentation(sys.argv)
        time[1] += 15

    reg = False
    if "-reg" in sys.argv[1:]:
        reg = True
        time[0] += 1
        time[1] += 45

    tract = False
    if "-tract" in sys.argv[1:]:
        tract = True
        time[0] += 1
        time[1] += 15

    time[0] += time[1]//60
    time[1] %= 60

    ## Read the list of subjects and for each subject do the tractography
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as file:
        patient_list = json.load(file)

    job_list = []

    for p_code in patient_list:
        p_job = {
            "wrap" : "export MKL_NUM_THREADS=4 ; export OMP_NUM_THREADS=4 ; python -c 'from tractography import compute_tracts; compute_tracts(\"%s\", \"%s\", %s, \"%s\", %s, %s)'" % (p_code, folder_path, str(extract_roi), seg_path, str(reg), str(tract)),
            "job_name" : "TRACKING_" + p_code,
            "ntasks" : 1,
            "cpus_per_task" : 4,
            "mem_per_cpu" : 2048,
            "time" : "%s:%s:00" % (str(time[0]), str(time[1])),
            "mail_user" : "michele.cerra@student.uclouvain.be",
            "mail_type" : "FAIL",
            "output" : folder_path + '/subjects/' + p_code + "/dMRI/tractography/slurm-%j.out",
            "error" : folder_path + '/subjects/' + p_code + "/dMRI/tractography/slurm-%j.err",
        }
        p_job_id = {}
        p_job_id["id"] = submit_job(p_job)
        p_job_id["name"] = p_code
        job_list.append(p_job_id)
    
    elikopy.utils.getJobsState(folder_path, job_list, "log")

if __name__ == "__main__":
    exit(main())