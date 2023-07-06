import sys
import os
import elikopy
import json

from params import *
from elikopy.utils import submit_job

absolute_path = os.path.dirname(__file__) # return the abs path of the folder of this file, wherever it is


def main():

    ## Getting input tracts
    input_path = get_inputTract(sys.argv)

    ## Getting folder
    folder_path = get_folder(sys.argv)

    if "-p" in sys.argv[1:]:
        parIdx = sys.argv.index("-p") + 1 # the index of the parameter after the option
        pat = sys.argv[parIdx]

    time = [0, 1]

    compute_5tt = False
    if "-5tt" in sys.argv[1:]:
        compute_5tt = True
        time[1] += 30

    extract_roi = False
    if "-roi" in sys.argv[1:]:
        extract_roi = True  
        time[1] += 30

    tract = False
    if "-tract" in sys.argv[1:]:    
        tract = True
        time[0] += 5
        time[1] += 0
    
    force = False
    if "-force" in sys.argv[1:]:
        force = True

    time[0] += time[1]//60
    time[1] %= 60

    ## Read the list of subjects and for each subject do the tractography
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as file:
        patient_list = json.load(file)

    if "-p" in sys.argv[1:]:
        if pat in patient_list:
            patient_list = [pat]
        else:
            print("Error: The inserted patient doesn't exist")
            exit(1)

    side = ""
    if "-side" in sys.argv[1:]:
        parIdx = sys.argv.index("-side") + 1 # the index of the parameter after the option
        side = sys.argv[parIdx]


    # check if the user wants to compute the ODF and compute it
    if "-odf" in sys.argv[1:]: 
        study = elikopy.core.Elikopy(folder_path, cuda=False, slurm=True, slurm_email="michele.cerra@student.uclouvain.be")

        study.odf_msmtcsd(patient_list_m=patient_list)
        return 0
    
    job_list = []

    for p_code in patient_list:
        if not os.path.isdir(folder_path + '/subjects/' + p_code + "/dMRI/tractography/"):
            os.mkdir(folder_path + '/subjects/' + p_code + "/dMRI/tractography/")

        p_job = {
            "wrap" : "export MKL_NUM_THREADS=4 ; export OMP_NUM_THREADS=4 ; python -c 'from tractography import compute_tracts; compute_tracts(\"%s\", \"%s\", \"%s\", %s, %s, %s, %s, \"%s\")'" % (input_path, p_code, folder_path, str(compute_5tt), str(extract_roi), str(tract), str(force), side),
            "job_name" :  p_code,
            "ntasks" : 1,
            "cpus_per_task" : 4,
            "mem_per_cpu" : 1024,
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