import sys
import os
import elikopy

from elikopy.core import Elikopy

def preprocess(folder_path, slurm=False):

    ## Connect
    study : Elikopy = elikopy.core.Elikopy(folder_path, cuda=True, slurm=slurm) # Set cuda to True if you want to use cuda for eddy current computation
    study.patient_list() 
    #  Create the subject folder with a .json
    #  in which are listed all the subject.
    #  It will be usefull later.

    # Preprocessing
    # Set the parameters depending on your volumes, see documentation...
    study.preproc(
    
        # Brain Extraction (Default Values)
    
        ## MPPCA denoising
        denoising=True,
    
        ## GIBBS Ringing Correction
        gibbs=False,
    
        ## Susceptibility field estimation
        topup=True,
        forceSynb0DisCo=True,
    
        ## EDDY and MOTION correction
        eddy=True,
        cuda_name="eddy_cuda10.2", # if you are using cuda check the eddy_cuda version you have installed. My case is 10.2
    
        ## BIAS FIELD correction
        biasfield=True,
    
        ## Quality check registration
        qc_reg=False,

        ## number of cpus
        cpus=8,
    )

    # Not always this command works
    study.white_mask(
        maskType="wm_mask_FSL_T1",
    )
    
    study.odf_msmtcsd(folder_path) # Distributions for tractography
    study.dti()
    study.noddi()

    # Unfortunatly, the DIAMOND code is not publicly available. If you do not have it in your possession, you will not be able to use this algorithm.

    #study.diamond()

    # Microstructure Fingerprinting can be computed only if you have a precomputed dictionary.
    # If you are in CECI cluster in the following path is present a dictionary.

    #dictionary_path = ".../ndelinte/fixed_rad_dist_wide.mat"

    # If you are not in CECI cluster some already computed dictionary are present in rensonnetg/microstructure_fingerprinting github repository in microstructure_fingerprinting/MCF_data.

    #dictionary_path = "...microstructure_fingerprinting/MCF_data/MCF.mat"
    
    #study.fingerprinting(dictionary_path=dictionary_path)

    return 0

def main():

    ## Getting Folder

    if "-f" in sys.argv[1:]:
        parIdx = sys.argv.index("-f") + 1 # the index of the parameter after the option
        par = sys.argv[parIdx]
        if os.path.isdir(par) and os.access(par,os.W_OK|os.R_OK):
            if par[-1] == "/":
                par = par[:-1]
            folder_path = par
        else:
            print("The inserted path doesn't exist or you don't have the access")
            exit(1)
    else:
        print("The folder path for segmentation isn't defined")
        exit(1)

    onCECI = False
    if "-CECI" in sys.argv[1:]:
        parIdx = sys.argv.index("-CECI") + 1 # the index of the parameter after the option
        par = sys.argv[parIdx]
        onCECI = par

    preprocess(folder_path, slurm=onCECI)


if __name__ == "__main__":
    exit(main())
