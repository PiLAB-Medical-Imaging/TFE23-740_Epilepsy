import sys
import elikopy

from params import *
from elikopy.utils import submit_job
from elikopy.core import Elikopy

def main():

    ## Getting Folder
    folder_path = get_folder(sys.argv)

    job = {
            "wrap" : "export MKL_NUM_THREADS=1 ; export OMP_NUM_THREADS=1 ; python -c 'from preproc import preprocess; preprocess(\"%s\")'" % (folder_path),
            "job_name" : "PREPROC",
            "ntasks" : 1,
            "cpus_per_task" : 1,
            "mem_per_cpu" : 512,
            "time" : "48:00:00",
            "mail_user" : "michele.cerra@student.uclouvain.be",
            "mail_type" : "FAIL",
            "output" : folder_path + "/subjects/slurm-%j.out",
            "error" : folder_path + "/subjects/slurm-%j.err",
        }
    job_id = {}
    job_id["id"] = submit_job(job)
    job_id["name"] = "master"
    
    elikopy.utils.getJobsState(folder_path, [job_id], "log")


if __name__ == "__main__":
    exit(main())