import os

def get_arguments(argv):
    ## Defining Parameters
    onSlurm = False
    slurmEmail = None
    cuda = False
    folder_path = None

    if len(argv) < 5 :
        print("ERROR: not enough parameters\n -w [lemaitre3|manneback|own] -f [directory_path]")
        exit(1)

    if "-w" in argv[1:]:
        parIdx = argv.index("-w") + 1 # the index of the parameter after the option
        par = argv[parIdx]
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

    return (onSlurm, slurmEmail, cuda, folder_path)