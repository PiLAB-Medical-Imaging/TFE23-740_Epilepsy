import sys
import json 

from params import get_fold
from elikopy.utils import submit_job

def main():
    study_fold = get_fold(sys.argv, "f")

    step = ""
    if "-prep" in sys.argv[1:]:
        step = "prep"
        time = 10
    if "-prior" in sys.argv[1:]:
        step = "prior"
        time = 5
    elif "-bedp" in sys.argv[1:]:
        step = "bedp"
        time = 48
    elif "-path" in sys.argv[1:]:
        step = "path"
        time = 24
    elif "-stat" in sys.argv[1:]:
        step = "stat"
        time = 24

    ## Read the list of subjects and for each subject do the tractography
    dest_success = study_fold + "/freesurfer/subj_list.json"
    with open(dest_success, 'r') as file:
        patient_list = json.load(file)

    if "-p" in sys.argv[1:]:
        parIdx = sys.argv.index("-p") + 1 # the index of the parameter after the option
        pat = sys.argv[parIdx]
        if pat in patient_list:
            patient_list = [pat]
        else:
            print("Error: The inserted patient doesn't exist")
            exit(1)

    job_list = []
    
    for p in patient_list:
        confFile = open("%s/freesurfer/confTrac/%s_tracula.conf" % (study_fold, p), "w")
        print("%s/freesurfer/confTrac/%s_tracula.conf" % (study_fold, p))
        confFile.write("setenv SUBJECTS_DIR %s/freesurfer/\n" % study_fold)
        confFile.write("set dtroot = %s/freesurfer/\n" % study_fold)
        confFile.write("set subjlist = (%s)\n" % p)
        
        # confFile.write("set dcmlist = (%s/subjects/%s/dMRI/preproc/%s_dmri_preproc.nii.gz)\n" % (study_fold, p, p))
        # confFile.write("set bveclist = (%s/subjects/%s/dMRI/preproc/%s_dmri_preproc.bvec)\n" % (study_fold, p, p))
        # confFile.write("set bvallist = (%s/subjects/%s/dMRI/preproc/%s_dmri_preproc.bval)\n" % (study_fold, p, p))
        
        confFile.write("set dcmlist = (%s/data_1/%s.nii.gz)\n" % (study_fold, p))
        confFile.write("set bveclist = (%s/data_1/%s.bvec)\n" % (study_fold, p))
        confFile.write("set bvallist = (%s/data_1/%s.bval)\n" % (study_fold, p))
        
        confFile.write("set doeddy = 0\n")
        confFile.write("set usethalnuc = 1\n")
        confFile.close()

        p_job = {
            "wrap" : "rm %s/freesurfer/%s/scripts/IsRunning.trac ; trac-all -c %s/freesurfer/confTrac/%s_tracula.conf -%s" % (study_fold, p, study_fold, p, step),
            "job_name" : "Tracu_" + p,
            "ntasks" : 1,
            "cpus_per_task" : 1,
            "mem_per_cpu" : 5120,
            "time" : "%d:00:00" % time,
            "mail_user" : "michele.cerra@student.uclouvain.be",
            "mail_type" : "FAIL",
            "output" : study_fold + f"/freesurfer/slurm-%j.out",
            "error" : study_fold + f"/freesurfer/slurm-%j.err",
        }
        p_job_id = {}
        p_job_id["id"] = submit_job(p_job)
        p_job_id["name"] = p
        job_list.append(p_job_id)

if __name__ == "__main__":
    exit(main())
