import sys
import os
import elikopy
import json

from params import get_folder
from elikopy.utils import submit_job

def main():

    ## Getting folder
    folder_path = get_folder(sys.argv)

    time = [1, 30]

    time[0] += time[1]//60
    time[1] %= 60

    ## Read the list of subjects and for each subject do the tractography
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as file:
        patient_list = json.load(file)

    job_list = []

    for p_code in patient_list:
        p_job = {
            "wrap" : "export MKL_NUM_THREADS=1 ; export OMP_NUM_THREADS=1 ; python -c 'from metrics import compute_metricsPerROI; compute_metricsPerROI(\"%s\", \"%s\")'" % (p_code, folder_path),
            "job_name" :  p_code,
            "ntasks" : 1,
            "cpus_per_task" : 1,
            "mem_per_cpu" : 1024,
            "time" : "%s:%s:00" % (str(time[0]), str(time[1])),
            "mail_user" : "michele.cerra@student.uclouvain.be",
            "mail_type" : "FAIL",
            "output" : folder_path + '/subjects/' + p_code + "/dMRI/microstructure/slurm-%j.out",
            "error" : folder_path + '/subjects/' + p_code + "/dMRI/microstructure/slurm-%j.err",
        }
        p_job_id = {}
        p_job_id["id"] = submit_job(p_job)
        p_job_id["name"] = p_code
        job_list.append(p_job_id)
    
    elikopy.utils.getJobsState(folder_path, job_list, "log")

if __name__ == "__main__":
    exit(main())