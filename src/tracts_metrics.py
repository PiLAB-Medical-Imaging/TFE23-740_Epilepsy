import os
import pandas as pd
import numpy as np
import nibabel as nib

from dipy.io.streamline import load_tractogram
from unravel.utils import *

def dti(trk, dti_map):
    density_map = get_streamline_density(trk)

    weightedMean = np.sum(density_map*dti_map)/np.sum(density_map)
    print(weightedMean)

def noddi():
    pass

def diamond():
    pass

def mf():
    # TODO read the paper of mf and if you still have questions go to nicolas
    pass

metrics = {
    dti : ["FA", "AD", "RD", "MD"],
    mf : ["fvf_f0", "fvf_f1"]
}

trks = {}
   
def tract_metrics(p_code, folder_path):
    subject_path = "%s/subjects/%s" % (folder_path, p_code)
    tracts_path = "%s/dMRI/tractography" % subject_path
    m = {}

    for model, m_metrics in metrics.items():
        model_path = "%s/dMRI/microstructure/%s" % (subject_path, model.__name__)
        if not os.path.isdir(model_path):
            print("Model metrics don't exist")

        for metric in m_metrics:
            metric_path = None
            if model.__name__ == "dti":
                metric_path = "%s/%s_%s.nii.gz" % (model_path, p_code, metric)
            else:
                metric_path = "%s/%s_%s_%s.nii.gz" % (model_path, p_code, model.__name__, metric)

            metric_map = nib.load(metric_path).get_fdata()

            for tract_filename in os.listdir(tracts_path):
                tract_name, ext = tract_filename.split(".")
                if ext == ".trk":
                    tract_path = os.path.join(tract_path, tract_filename)

                    # save in memory the trackts, in order to not open them every time
                    if tract_path not in trks:
                        trk = load_tractogram(tract_path)
                        trk.to_vox()
                        trk.to_corner()
                        trks[tract_path] = trk
                    else:
                        trk = trks[tract_path]

                    attr_name = "%s_%s" % (tract_name, metric)
                    m[attr_name] = model(trk, metric_map)
