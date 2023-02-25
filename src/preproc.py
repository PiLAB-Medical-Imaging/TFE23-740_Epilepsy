import sys
import os

import elikopy
import elikopy.utils

def main():
    ## Defining Parameters
    onSlurm = False
    slurmEmail = None
    starting_state = None
    cuda = False
    f_path=os.getcwd()

    if len(sys.argv) == 1 :
        print("ERROR: not enough parameters\n -w [lemaitre3|manneback|own] -f [directory_path] -s [starting_state]")
        exit(1)

    if "-w" in sys.argv[1:]:
        parIdx = sys.argv.index("-w") + 1 # the index of the parameter after the option
        par = sys.argv[parIdx]
        if par in ["lemaitre3", "manneback"]:
            onSlurm = True; slurmEmail="fbmichele@hotmail.it"
            if par == "manneback":
                cuda = True
            print("Running on CECI cluster")
        elif par == "own":
            print("Running on this terminal")
            pass
        else:
            print("Unknown position where run the script")
            exit(1)
    else:
        print("None position is defined\n -w [ceci|own]")
        exit(1)
    
    if "-f" in sys.argv[1:]:
        parIdx = sys.argv.index("-f") + 1 # the index of the parameter after the option
        par = sys.argv[parIdx]
        if os.path.isdir(par) and os.access(par,os.W_OK|os.R_OK):
            if par[-1] == "/":
                par = par[:-1]
            f_path = par
        else:
            print("The inserted path doesn't exist or you don't have the access")
            exit(1)
    else:
        print("The folder path isn't defined: It will be used \"%s\"" % f_path)

    if "-s" in sys.argv[1:]:
        parIdx = sys.argv.index["-s"] + 1 # the index of the parameter after the option
        par = sys.argv[parIdx]
        assert par in (None, "None", "denoising", "gibbs", "topup", "eddy", "biasfield", "report", "topup_synb0DisCo_Registration", "topup_synb0DisCo_Inference", "topup_synb0DisCo_Apply", "topup_synb0DisCo_topup"), 'invalid starting state!'
        starting_state = par

    ## Init
    study = elikopy.core.Elikopy(f_path, cuda=cuda, slurm=onSlurm, slurm_email=slurmEmail)
    study.patient_list() #TODO Check if our volumes are acquired in reverse phase encoding

    ## Preprocessing
    ## RESLICING & BET
    study.preproc(
        # Starting STATE
        starting_state=starting_state,

        # Reslicing
        reslice=True, #TODO chiedere se le nostre immagini sono interpolate, perche se non lo sono questo può essere anche a False
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
        topupConfig=None, #TODO What is it?
        forceSynb0DisCo=True,

        ## EDDY and MOTION correction
        #TODO Studiare come si generano gli artefatti
        #TODO Capire cosa fa FSL per correggere eddy currents e outlier replacement
        eddy=True,
        #niter=,
        #s2v=, # Doing only the Volume-to-Volume preproc is good enough
        cuda=cuda,
        cuda_name="eddy_cuda10.2", # Depends on the version that you have installed
        #olrep=,

        # Registration
        qc_reg=False,

        ## BIAS FIELD correction
        biasfield=True,
        #biasfield_bsplineFitting=,
        #biasfield_convergence=,
    )

    return 0


if __name__ == "__main__":
    exit(main())
