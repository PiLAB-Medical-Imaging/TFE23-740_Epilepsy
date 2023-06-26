import sys
import json 

from params import get_fold
from elikopy.utils import submit_job

def main():
    study_fold = get_fold(sys.argv, "f")

    step = ""
    if "-prep" in sys.argv[1:]:
        step = "prep"
    elif "-bedp" in sys.argv[1:]:
        step = "bedp"
    elif "-path" in sys.argv[1:]:
        step = "path"
    elif "-stat" in sys.argv[1:]:
        step = "stat"

    ## Read the list of subjects and for each subject do the tractography
    dest_success = study_fold + "/freesurfer/subj_list.json"
    with open(dest_success, 'r') as file:
        patient_list = json.load(file)

    job_list = []
    
    for p in patient_list:
        confFile = open("%s/freesurfer/confTrac/%s_tracula.conf" % (study_fold, p), "w")
        print("%s/freesurfer/confTrac/%s_tracula.conf" % (study_fold, p))
        confFile.write("setenv SUBJECTS_DIR %s/freesurfer/\n" % study_fold)
        confFile.write("set dtroot = %s/freesurfer/\n" % study_fold)
        confFile.write("set subjlist = (%s)\n" % p)
        confFile.write("set dcmroot = %s/subjects/%s/dMRI/preproc/\n" % (study_fold, p))
        confFile.write("set dcmlist = (%s_dmri_preproc.nii.gz)\n" % p)
        confFile.write("set bveclist = (%s/subjects/%s/dMRI/preproc/%s_dmri_preproc.bvec)\n" % (study_fold, p, p))
        confFile.write("set bvallist = (%s/subjects/%s/dMRI/preproc/%s_dmri_preproc.bval)\n" % (study_fold, p, p))
        confFile.write("set doeddy = 0\n")
        confFile.close()

        p_job = {
            "wrap" : "trac-all -c %s/freesurfer/confTrac/%s_tracula.conf -%s" % (study_fold, p, step),
            "job_name" : "Tracu_" + p,
            "ntasks" : 1,
            "cpus_per_task" : 2,
            "mem_per_cpu" : 5120,
            "time" : "24:00:00",
            "mail_user" : "michele.cerra@student.uclouvain.be",
            "mail_type" : "FAIL",
            "output" : study_fold + "/freesurfer/slurm-%j.out",
            "error" : study_fold + "/freesurfer/slurm-%j.err",
        }
        p_job_id = {}
        p_job_id["id"] = submit_job(p_job)
        p_job_id["name"] = p
        job_list.append(p_job_id)

if __name__ == "__main__":
    exit(main())