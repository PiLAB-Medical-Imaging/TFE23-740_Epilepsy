import sys

import elikopy
import elikopy.utils
from params import get_folder, get_model
from elikopy.core import Elikopy

models = ["all", "dti", "noddi", "diamond", "mf"]

def main():
    ## Getting Parameters
    folder_path = get_folder(sys.argv)
    model = None

    dictionary_path = "/home/users/n/d/ndelinte/fixed_rad_dist_wide.mat" # taken from the code of Alexandre

    ## Define which metric want to estimate
    model = get_model(sys.argv)

    ## Connect
    study : Elikopy = elikopy.core.Elikopy(folder_path, cuda=True, slurm=True, slurm_email="michele.cerra@student.uclouvain.be")
    
    ## Metrics Estimation
    print("Computing %s model for all the subjects" % model.upper())
    if model in ("dti", "all"):
        study.dti()
    if model in ("diamond", "all"):
        study.diamond()
    if model in ("noddi", "all"):
        study.noddi()
    if model in ("mf", "all"):
        study.fingerprinting(
            dictionary_path=dictionary_path,
        )

if __name__ == "__main__":
    exit(main())
