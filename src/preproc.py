import sys
import os

import elikopy
import elikopy.utils

preproc_steps = [None, "None", "denoising", "gibbs", "topup", "eddy", "biasfield", "report", "topup_synb0DisCo_Registration", "topup_synb0DisCo_Inference", "topup_synb0DisCo_Apply", "topup_synb0DisCo_topup"]
models_steps = [None, "dti", "noddi", "diamond", "mf"]

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
        assert par in preproc_steps or par in models_steps or par in ("white_mask", "tracking"), 'invalid starting state!'
        starting_state = par

    ## Init
    study = elikopy.core.Elikopy(f_path, cuda=cuda, slurm=onSlurm, slurm_email=slurmEmail)
    study.patient_list() #TODO Check if our volumes are acquired in reverse phase encoding

    ## Preprocessing
    if starting_state in preproc_steps:
        study.preproc(
            # Starting STATE
            starting_state=starting_state,

            # Reslicing
            reslice=False, #TODO chiedere se le nostre immagini sono interpolate, perche se non lo sono questo può essere anche a False
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
            forceSynb0DisCo=True, # TODO I should do it? I didn't understand if only a single phase encoding is available

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
            qc_reg=False, #TODO Is done just for quality control, do at least one time to check

            ## BIAS FIELD correction
            biasfield=True,
            #biasfield_bsplineFitting=,
            #biasfield_convergence=,
        )
        starting_state = None # In this way it enter in the next steps.

    # """
    # Providing a white matter mask is a useful step to accelerate 
    # microstructural features computation and more easily do tractography.
    # """
    # if starting_state in ("white_mask", None):
    #     study.white_mask(
    #         maskType="wm_mask_FSL_T1",
    #         corr_gibbs=False, # I have to put false because there is a problem with the parameters in gibbs_removal
    #     )
    #     starting_state=None

    if starting_state in ("tracking", None):
        study.odf_msmtcsd()
        study.tracking()
        starting_state=None
    
    ## Metrics Estimation 
    
    if starting_state in models_steps:
        mask_to_use = "brain_mask_dilated"
        if starting_state in ("dti", None):
            study.dti(
                # mask to use
                maskType=mask_to_use,
                # if you want to other mask you have to compute it singularly with white_mask() 
            )
            starting_state = None
        if starting_state in ("diamond", None):
            study.diamond(
                # mask to use
                maskType=mask_to_use,
                # other params
                customDiamond="", #TODO add or change it
            )
            starting_state = None
        if starting_state in ("noddi", None):
            study.noddy(
                # mask to use
                maskType=mask_to_use,
                #lambda_iso_diff=,
                #lambda_par_diff=,
            )
            starting_state = None
        # if starting_state in ("mf", None):
        #     study.fingerprinting(
        #         #dictionary_path=,
        #         maskType=mask_to_use,
        #     )
      
    return 0


if __name__ == "__main__":
    exit(main())
