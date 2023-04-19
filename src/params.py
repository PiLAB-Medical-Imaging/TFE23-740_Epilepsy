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