import sys
import os
import elikopy

from elikopy.core import Elikopy
from elikopy.utils import submit_job

from params import *

def preprocess(folder_path):

    ## Connect
    study : Elikopy = elikopy.core.Elikopy(folder_path, cuda=False, slurm=True, slurm_email="michele.cerra@student.uclouvain.be")
    study.patient_list()

    ## Preprocessing
    study.preproc(

        # Reslicing
        reslice=False,
        reslice_addSlice=False, # Because we don't perform slice-to-volume corrections

        # Brain Extraction (Default Values)

        ## MPPCA denoising
        denoising=True,
        denoising_algorithm="mppca_mrtrix",

        ## GIBBS Ringing Correction
        gibbs=False,

        ## Susceptibility field estimation
        topup=True,
        forceSynb0DisCo=True,

        ## EDDY and MOTION correction
        eddy=True,
        cuda_name="eddy_cuda10.2", # Depends on the version that you have installed

        ## BIAS FIELD correction
        biasfield=True, 
    )
    return 0

def main():

    ## Getting Folder
    folder_path = get_folder(sys.argv)

    job = {
            "wrap" : "export MKL_NUM_THREADS=4 ; export OMP_NUM_THREADS=4 ; python -c 'from preproc import preprocess; preprocess(\"%s\")'" % (folder_path),
            "job_name" : "PREPROCM",
            "ntasks" : 1,
            "cpus_per_task" : 4,
            "mem_per_cpu" : 2048,
            "time" : "15:00:00",
            "mail_user" : "michele.cerra@student.uclouvain.be",
            "mail_type" : "FAIL",
        }
    job_id = {}
    job_id["id"] = submit_job(job)
    job_id["name"] = "master"
    
    elikopy.utils.getJobsState(folder_path, [job_id], "log")


if __name__ == "__main__":
    exit(main())
