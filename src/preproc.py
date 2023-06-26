import sys
import os
import elikopy

from elikopy.core import Elikopy
from elikopy.utils import submit_job

from params import *

def preprocess(folder_path, slurm=True):

    ## Connect
    study : Elikopy = elikopy.core.Elikopy(folder_path, cuda=False, slurm=slurm, slurm_email="michele.cerra@student.uclouvain.be")
    study.patient_list()

    ## Preprocessing
    study.preproc(

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
        dynamic_susceptibility=True,

        ## BIAS FIELD correction
        biasfield=True,

        ## Registration
        qc_reg=True,
    )

    study.white_mask(
        maskType="wm_mask_FSL_T1",
    )

    return 0

def main():

    ## Getting Folder
    folder_path = get_folder(sys.argv)
    
    preprocess(folder_path, slurm=False)


if __name__ == "__main__":
    exit(main())
