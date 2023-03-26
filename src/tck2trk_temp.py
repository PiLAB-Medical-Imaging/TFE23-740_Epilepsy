import sys
import os
from dipy.io.streamline import load_tractogram, save_trk

def main():
    if len(sys.argv) != 3 :
        print("ERROR: not enough parameters\n")
        exit(1)
    
    tck_path = sys.argv[1]
    dwi_path = sys.argv[2]

    # implementare direttamente questo nel bash..

    tract = load_tractogram(tck_path, dwi_path)

    save_trk(tract, tck_path[:-3]+'trk')

if __name__ == "__main__":
    exit(main())
