import sys

import elikopy
import elikopy.utils
from params import get_arguments

models_steps = ["all", "dti", "noddi", "diamond", "mf"]

def main():
    ## Getting Parameters
    onSlurm, slurmEmail, cuda, f_path = get_arguments(sys.argv)
    model = None

    dictionary_path = f_path + "/static_files/mf_dic/fixed_rad_dist_wide.mat" if onSlurm else "/home/users/n/d/ndelinte/fixed_rad_dist_wide.mat"

    ## Define which metric want to estimate
    if "-s" in sys.argv[1:]:
        parIdx = sys.argv.index("-s") + 1 # the index of the parameter after the option
        par = sys.argv[parIdx]
        assert par in models_steps, 'invalid model!'
        model = par
    else:
        print("no model selected!")
        exit(1)

    ## Connect
    study = elikopy.core.Elikopy(f_path, cuda=cuda, slurm=onSlurm, slurm_email=slurmEmail)
    
    ## Metrics Estimation
    if model in ("dti", "all"):
        study.dti(
            # if you want to other mask you have to compute it singularly with white_mask() 
        )
    if model in ("diamond", "all"):
        study.diamond(
            # other params
            customDiamond="", #TODO add or change it
        )
    if model in ("noddi", "all"):
        study.noddy(
            #lambda_iso_diff=,
            #lambda_par_diff=,
        )
    if model in ("mf", "all"):
        print("dic path: %s" % dictionary_path)

        study.fingerprinting(
            dictionary_path=dictionary_path,
        )

if __name__ == "__main__":
    exit(main())
