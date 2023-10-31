import sys
import elikopy

from elikopy.core import Elikopy

from params import *

def preprocess(folder_path, slurm=False):

    ## Connect
    study : Elikopy = elikopy.core.Elikopy(folder_path, cuda=False, slurm=slurm)
    study.patient_list()

    # Preprocessing
    # Set the parameters depending on your volumes, see documentation...
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
    
        ## Quality check registration
        qc_reg=False,
    )

    # Not always this command works
    study.white_mask(
        maskType="wm_mask_FSL_T1",
    )
    
    study.odf_msmtcsd(folder_path)
    study.dti()
    study.noddi()

    # Unfortunatly, the DIAMOND code is not publicly available. If you do not have it in your possession, you will not be able to use this algorithm.
    #study.diamond()

    # Microstructure Fingerprinting can be computed only if you have a precomputed dictionary.
    # If you are in CECI cluster in the following path is present a dictionary.
    #dictionary_path = "/home/users/n/d/ndelinte/fixed_rad_dist_wide.mat"
    #study.fingerprinting(dictionary_path=dictionary_path)

    return 0

def main():

    ## Getting Folder
    folder_path = get_folder(sys.argv)
    
    preprocess(folder_path, slurm=False) # Set True if you are working on CECI cluster


if __name__ == "__main__":
    exit(main())
