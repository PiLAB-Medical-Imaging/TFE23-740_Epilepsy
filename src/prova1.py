import sys
import json
import time

from params import get_folder, get_segmentation
from multiprocessing.pool import ThreadPool

nThreads = 2

def stampante(carattere1, carattere2):
    time.sleep(1)

    time.sleep(carattere2)
    print(carattere1, carattere2, "prima")
    

a = ["a", "b", "c", "d", "e", "f"]
b = [10, 5, 1, 1, 2, 2]

with ThreadPool(nThreads) as pool:
    args = [(a[i], b[i]) for i in range(len(a))]
    results = pool.starmap(stampante, args)
