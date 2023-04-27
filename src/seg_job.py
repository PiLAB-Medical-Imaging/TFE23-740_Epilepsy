import sys
import json
import elikopy

from params import get_segmentation, get_folder
from elikopy.utils import submit_job

# srun recon-all -all -s $SUB_ID -i $PROJECT_DIR/study/T1/${SUB_ID}_T1.nii.gz -T2 $PROJECT_DIR/study/T1/${SUB_ID}_T2.nii.gz -T2pial -qcache
# srun --cpus-per-task=4 mri_cc -force -f -aseg aseg.mgz -o aseg.auto_CCseg.mgz $SUB_ID

def main():
    folder_path = get_folder(sys.argv)
    seg_fold = get_segmentation(sys.argv)

    ## Read the list of subjects and for each subject do the tractography
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as file:
        patient_list = json.load(file)

    job_list = []

    for i in range(19,20): # CHANGE with patient list | all the patients
        p_job = {
            "wrap" : "recon-all -all -sd %s -s VNSLC_%02d -i %s/T1/VNSLC_%02d_T1.nii.gz -T2 %s/T1/VNSLC_%02d_T2.nii.gz -T2pial -qcache" % (seg_fold, i, folder_path, i, folder_path, i,),
            "job_name" : "Seg_" + str(i),
            "ntasks" : 1,
            "cpus_per_task" : 4,
            "mem_per_cpu" : 4096,
            "time" : "24:00:00",
            "mail_user" : "michele.cerra@student.uclouvain.be",
            "mail_type" : "FAIL",
            "output" : seg_fold + "/slurm-%j.out",
            "error" : seg_fold + "/slurm-%j.err",
        }
        p_job_id = {}
        p_job_id["id"] = submit_job(p_job)
        p_job_id["name"] = "VNSLC_%02d" % i
        job_list.append(p_job_id)
    
    elikopy.utils.getJobsState(folder_path, job_list, "log")

if __name__ == "__main__" :
    exit(main())