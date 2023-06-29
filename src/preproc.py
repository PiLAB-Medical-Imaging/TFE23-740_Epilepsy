import sys
import elikopy

from elikopy.core import Elikopy

from params import *

def preprocess(folder_path, slurm=True):

    ## Connect
    study : Elikopy = elikopy.core.Elikopy(folder_path, cuda=False, slurm=slurm, slurm_email="michele.cerra@student.uclouvain.be")
    study.patient_list()

    ## Preprocessing
    # study.preproc(
    # 
    #     # Brain Extraction (Default Values)
    # 
    #     ## MPPCA denoising
    #     denoising=True,
    #     denoising_algorithm="mppca_mrtrix",
    # 
    #     ## GIBBS Ringing Correction
    #     gibbs=False,
    # 
    #     ## Susceptibility field estimation
    #     topup=True,
    #     forceSynb0DisCo=True,
    # 
    #     ## EDDY and MOTION correction
    #     eddy=True,
    #     dynamic_susceptibility=True,
    # 
    #     ## BIAS FIELD correction
    #     biasfield=True,
    # 
    #     ## Quality check registration
    #     qc_reg=False,
    # )

    study.white_mask(
        maskType="wm_mask_FSL_T1",
    )

    # dictionary_path = "/home/users/n/d/ndelinte/fixed_rad_dist_wide.mat" # taken from the code of Alexandre
    # 
    # study.odf_msmtcsd(folder_path)
    # study.dti()
    # study.noddi()
    # study.diamond()
    # study.fingerprinting(dictionary_path=dictionary_path)

    return 0

def main():

    ## Getting Folder
    folder_path = get_folder(sys.argv)
    
    preprocess(folder_path, slurm=False)


if __name__ == "__main__":
    exit(main())
