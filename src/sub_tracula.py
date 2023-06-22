from elikopy.utils import submit_job

def main():

    p_job = {
        "wrap" : "trac-all -c ./study/confTracula.txt -prep",
        "job_name" : "Trac02",
        "ntasks" : 1,
        "cpus_per_task" : 2,
        "mem_per_cpu" : 5120,
        "time" : "24:00:00",
        "mail_user" : "michele.cerra@student.uclouvain.be",
        "mail_type" : "FAIL",
        "output" : "./study/tracula/slurm-%j.out",
        "error" : "./study/tracula/slurm-%j.err",
    }
    submit_job(p_job)

if __name__ == "__main__" :
    exit(main())
