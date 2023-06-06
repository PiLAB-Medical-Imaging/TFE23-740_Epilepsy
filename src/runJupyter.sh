srun -J kernJupy -c 8 -n 1 -t 03:30:00 --mem-per-cpu 1024 --cluster=$CLUSTER_NAME --pty bash -c 'jupyter notebook --ip $(hostname -i) --no-browser'
