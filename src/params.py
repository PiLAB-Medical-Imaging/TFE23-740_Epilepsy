import os

def get_folder(argv):
    ## Defining Parameters
    onSlurm = False
    slurmEmail = None
    cuda = False
    folder_path = None
        
    if "-f" in argv[1:]:
        parIdx = argv.index("-f") + 1 # the index of the parameter after the option
        par = argv[parIdx]
        if os.path.isdir(par) and os.access(par,os.W_OK|os.R_OK):
            if par[-1] == "/":
                par = par[:-1]
            folder_path = par
        else:
            print("The inserted path doesn't exist or you don't have the access")
            exit(1)
    else:
        print("The folder path isn't defined")
        exit(1)

    return folder_path

def get_segmentation(argv):
    seg_fold = None

    if "-s" in argv[1:]:
        parIdx = argv.index("-s") + 1 # the index of the parameter after the option
        par = argv[parIdx]
        if os.path.isdir(par) and os.access(par,os.W_OK|os.R_OK):
            if par[-1] == "/":
                par = par[:-1]
            seg_fold = par
        else:
            print("The inserted path doesn't exist or you don't have the access")
            exit(1)
    else:
        print("The folder path for segmentation isn't defined")
        exit(1)
    
    return seg_fold

models = ["all", "dti", "noddi", "diamond", "mf"]
def get_model(argv):
    model = None
    if "-m" in argv[1:]:
        parIdx = argv.index("-m") + 1 # the index of the parameter after the option
        par = argv[parIdx]
        assert par in models, 'invalid model!'
        model = par
    else:
        print("no model selected!")
        exit(1)

    return model

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
