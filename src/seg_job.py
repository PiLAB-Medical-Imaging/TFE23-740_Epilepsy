# THIS MINI PROGRAM RUN THE SEGMENTATION FOR ALL THE PATIENTS
import subprocess
import os
absolute_path = os.path.dirname(__file__) # return the abs path of the folder of this file, wherever it is
project_dir = absolute_path + "/.."
subjects_dir = project_dir + "/seg_subjs"

nPatients = 23

# srun recon-all -all -s $SUB_ID -i $PROJECT_DIR/study/T1/${SUB_ID}_T1.nii.gz -T2 $PROJECT_DIR/study/T1/${SUB_ID}_T2.nii.gz -T2pial -qcache
# srun --cpus-per-task=4 mri_cc -force -f -aseg aseg.mgz -o aseg.auto_CCseg.mgz $SUB_ID

processes = []

for i in range(1,nPatients+1):
    bashCommand = "recon-all -all -sd %s -s VNSLC_%02d -i %s/study/T1/VNSLC_%02d_T1.nii.gz -T2 %s/study/T1/VNSLC_%02d_T2.nii.gz -T2pial -qcache > %s/outputs/seg_out_VNSLC_%02d.txt" % (subjects_dir, i, project_dir, i, project_dir, i, absolute_path, i)
    process = subprocess.Popen(bashCommand, universal_newlines=True, shell=True)
    processes.append(process)
    print("VNSLC_%02d" % i)

for process in processes:
    print("waiting")
    process.communicate()
