# THIS MINI PROGRAM RUN THE SEGMENTATION FOR ALL THE PATIENTS
import subprocess
import os
absolute_path = os.path.dirname(__file__) # return the abs path of the folder of this file, wherever it is

nPatients = 23

for i in range(1,nPatients+1):
    sub = "VNSLC_%02d" % i
    bashCommand = absolute_path + "./seg_job.sh " + sub
    process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True)