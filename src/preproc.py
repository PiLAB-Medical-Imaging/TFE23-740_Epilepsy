import sys
import os

import elikopy
import elikopy.utils

from params import get_arguments

def main():
    ## Getting Parameters
    onSlurm, slurmEmail, cuda, f_path = get_arguments(sys.argv)

    ## Init
    study = elikopy.core.Elikopy(f_path, cuda=cuda, slurm=onSlurm, slurm_email=slurmEmail)
    study.patient_list()

    ## Preprocessing
    study.preproc(

        # Reslicing
        reslice=False,
        reslice_addSlice=False, # Because we don't perform slice-to-volume corrections

        # Brain Extraction
        #bet_median_radius=,
        #bet_numpass=,
        #bet_dilate=,

        ## MPPCA denoising
        denoising=True,

        ## GIBBS Ringing Correction
        gibbs=False,

        ## Susceptibility field estimation
        topup=True,
        topupConfig=None, 
        forceSynb0DisCo=True,

        ## EDDY and MOTION correction
        eddy=True,
        #niter=,
        #s2v=, # Doing only the Volume-to-Volume preproc is good enough
        cuda=cuda,
        cuda_name="eddy_cuda10.2", # Depends on the version that you have installed
        #olrep=,

        # Registration
        qc_reg=False, # Is done just for quality control, do at least one time to check

        ## BIAS FIELD correction
        biasfield=True, # TODO should I put to false?
        #biasfield_bsplineFitting=,
        #biasfield_convergence=,
    )
    return 0

if __name__ == "__main__":
    exit(main())
