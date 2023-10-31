import os

def get_inputTract(argv):
    ## Defining Parameters
    input_path = None
        
    if "-i" in argv[1:]:
        parIdx = argv.index("-i") + 1 # the index of the parameter after the option
        par = argv[parIdx]
        if os.path.isfile(par) and os.access(par, os.R_OK) and par.endswith(".json"):
            if par[-1] == "/":
                par = par[:-1]
            input_path = par
        else:
            print("The inserted path doesn't exist or you don't have the access")
            exit(1)
    else:
        print("The input path isn't defined")
        exit(1)

    return input_path

def get_patient(argv):
    patient = None
    if "-p" in argv[1:]:
        parIdx = argv.index("-p") + 1 # the index of the parameter after the option
        par = argv[parIdx]
        patient = par
    else:
        print("no patient selected!")
        exit(1)

    return patient

def get_fold(argv, char_key):
    path_fold = None

    if "-"+char_key in argv[1:]:
        parIdx = argv.index("-"+char_key) + 1 # the index of the parameter after the option
        par = argv[parIdx]
        if os.path.isdir(par) and os.access(par,os.W_OK|os.R_OK):
            if par[-1] == "/":
                par = par[:-1]
            path_fold = par
        else:
            print("The inserted path doesn't exist or you don't have the access")
            exit(1)
    else:
        print("The folder path for segmentation isn't defined")
        exit(1)
    
    return path_fold

def get_onCECI(argv):
    onCECI = False
    if "-CECI" in argv[1:]:
        parIdx = argv.index("-CECI") + 1 # the index of the parameter after the option
        par = argv[parIdx]
        onCECI = par
    return onCECI
