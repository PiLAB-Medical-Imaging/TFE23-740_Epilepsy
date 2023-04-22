import sys
import json
import time

from params import get_folder, get_segmentation
from multiprocessing.pool import ThreadPool

folder_path = get_folder(sys.argv)

dest_success = folder_path + "/subjects/subj_list.json"
with open(dest_success, 'r') as file:
    patient_list = json.load(file)

nThreads = 2

def stampante(carattere1, carattere2):
    time.sleep(1)

    time.sleep(carattere2)
    print(carattere1, carattere2)
    

a = ["a", "b", "c", "d", "e", "f"]
b = [2, 1, 2, 1, 2, 1]

with ThreadPool(nThreads) as pool:
    args = [(a[i], b[i]) for i in range(5)]
    results = pool.starmap(stampante, args)
