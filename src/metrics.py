import sys

import elikopy
import elikopy.utils
from params import get_folder, get_model

models = ["all", "dti", "noddi", "diamond", "mf"]

def main():
    ## Getting Parameters
    folder_path = get_folder(sys.argv)
    model = None

    dictionary_path = "/home/users/n/d/ndelinte/fixed_rad_dist_wide.mat"

    ## Define which metric want to estimate
    model = get_model(sys.argv)

    ## Connect
    study = elikopy.core.Elikopy(folder_path, cuda=True, slurm=True, slurm_email="michele.cerra@student.uclouvain.be")
    
    ## Metrics Estimation
    print("Computing %s model for all the subjects" % model.upper())
    if model in ("dti", "all"):
        study.dti()
    if model in ("diamond", "all"):
        study.diamond(
            # other params
            customDiamond="", #TODO add or change it
        )
    if model in ("noddi", "all"):
        study.noddi_amico(
            #lambda_iso_diff=,
            #lambda_par_diff=,
        )
    if model in ("mf", "all"):
        study.fingerprinting(
            dictionary_path=dictionary_path,
        )

if __name__ == "__main__":
    exit(main())
