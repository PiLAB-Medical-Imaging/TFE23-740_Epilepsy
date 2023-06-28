import sys
import json
import elikopy

from params import get_fold
from elikopy.utils import submit_job

def main():
    study_fold = get_fold(sys.argv, "f")

    ## Read the list of subjects and for each subject do the tractography
    dest_success = study_fold + "/freesurfer/subj_list.json"
    with open(dest_success, 'r') as file:
        patient_list = json.load(file)

    #patient_list =  ["VNSLC_19"]

    job_list = []

    for p_name in patient_list:
        p_job = {
            "wrap" : "recon-all -all -sd %s/freesurfer/ -s %s -i %s/T1/%s_T1.nii.gz -T2 %s/T1/%s_T2.nii.gz -T2pial -qcache" % (study_fold, p_name, study_fold, p_name, study_fold, p_name),
            #"wrap" : "recon-all -all -sd %s/freesurfer/ -s %s -T2 %s/T1/%s_T2.nii.gz -T2pial -qcache" % (study_fold, p_name, study_fold, p_name),

            "job_name" : "Seg_" + p_name,
            "ntasks" : 1,
            "cpus_per_task" : 1,
            "mem_per_cpu" : 4096,
            "time" : "18:00:00",
            "mail_user" : "michele.cerra@student.uclouvain.be",
            "mail_type" : "FAIL",
            "output" : study_fold + "/freesurfer/slurm-%j.out",
            "error" : study_fold + "/freesurfer/slurm-%j.err",
        }
        p_job_id = {}
        p_job_id["id"] = submit_job(p_job)
        p_job_id["name"] = p_name
        job_list.append(p_job_id)
    
    elikopy.utils.getJobsState(study_fold, job_list, "log")

if __name__ == "__main__" :
    exit(main())
