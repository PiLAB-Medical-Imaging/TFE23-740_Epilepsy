import sys
import os
import elikopy

from elikopy.core import Elikopy

from params import get_arguments

def main():
    ## Getting Parameters
    onSlurm, slurmEmail, cuda, f_path = get_arguments(sys.argv)

    ## Init
    study : Elikopy = elikopy.core.Elikopy(f_path, cuda=cuda, slurm=onSlurm, slurm_email=slurmEmail)
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
        topupConfig=None, 
        forceSynb0DisCo=True,

        ## EDDY and MOTION correction
        eddy=True,

        ## BIAS FIELD correction
        biasfield=True, 
    )
    return 0

if __name__ == "__main__":
    exit(main())
