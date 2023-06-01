import sys
import os
import elikopy
import json

from params import get_folder
from elikopy.utils import submit_job

absolute_path = os.path.dirname(__file__) # return the abs path of the folder of this file, wherever it is


def main():

    ## Getting folder
    folder_path = get_folder(sys.argv)

    time = [1, 0]

    time[0] += time[1]//60
    time[1] %= 60

    job_list = []

    p_job = {
        "wrap" : "export MKL_NUM_THREADS=24 ; export OMP_NUM_THREADS=24 ; python -c 'from model_selection import model_selection; model_selection(\"%s\")'" % (folder_path),
        "job_name" :  "mod_sel",
        "ntasks" : 1,
        "cpus_per_task" : 4,
        "mem_per_cpu" : 256,
        "time" : "%s:%s:00" % (str(time[0]), str(time[1])),
        "mail_user" : "michele.cerra@student.uclouvain.be",
        "mail_type" : "FAIL",
        "output" : folder_path + "/stats/slurm-%j.out",
        "error" : folder_path + "/stats/slurm-%j.err",
    }
    p_job_id = {}
    p_job_id["id"] = submit_job(p_job)
    p_job_id["name"] = "mod_sel"
    job_list.append(p_job_id)
    
    elikopy.utils.getJobsState(folder_path, job_list, "log")

if __name__ == "__main__":
    exit(main())