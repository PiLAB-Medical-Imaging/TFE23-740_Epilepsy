import os
import pandas as pd
import numpy as np
import nibabel as nib

from dipy.io.streamline import load_tractogram
from unravel.utils import *

# for each compute the mean, std ,

def vcol(v):
    return v.reshape(v.size, 1)

def vrow(v):
    return v.reshape(1, v.size)

def w_mean(v, w):
    v = vcol(v)
    w = vrow(w)
    return np.dot(w,v)/np.sum(w)

def w_var(v, w):
    v = vcol(v)
    w = vrow(w)

    v_centr = v - w_mean(v, w) # centerinig the values

    V1 = np.sum(w**1)
    V2 = np.sum(w**2)

    w_var = np.dot(w, v_centr**2)/V1 # m_2

    w_var_unbias = V1**2/(V1**2-V2) * w_var # M_2

    return w_var_unbias

def w_skew(v, w):
    v = vcol(v)
    w = vrow(w)

    v_centr = v - w_mean(v, w) # centerinig the values

    V1 = np.sum(w**1)
    V2 = np.sum(w**2)
    V3 = np.sum(w**3)

    w_skew = np.dot(w, v_centr**3)/V1 # m_3

    w_skew_unbias = V1**3/(V1**3-3*V1*V2+2*V3) * w_skew # M_3

    return w_skew_unbias

def w_kurt(v, w):
    v = vcol(v)
    w = vrow(w)

    v_centr = v - w_mean(v, w) # centerinig the values

    V1 = np.sum(w**1)
    V2 = np.sum(w**2)
    V3 = np.sum(w**3)
    V4 = np.sum(w**4)

    w_var = np.dot(w, v_centr**2)/V1 # m_2
    w_kurt = np.dot(w, v_centr**4)/V1 # m_4

    denominator = (V1**2 - V2)*(V1**4 - 6*V1**2*V2 + 8*V1*V3 + 3*V2**2 - 6*V4)
    w_kurt_unbias = V1**2*(V1**4 - 3*V1**2*V2 + 2*V1*V3 + 3*V2**2 - 3*V4)/denominator * w_kurt - 3*V1**2*(2*V1**2*V2 - 2*V1*V3 - 3*V2**2 + 3*V4)/denominator * w_var**2

    return w_kurt_unbias


def dti(density_map, dti_map):

    weightedMean = w_mean(dti_map, density_map)
    weightedStd = np.sqrt(w_var(dti_map, density_map))
    weightedSkew = w_skew(dti_map, density_map)
    weightedKurt = w_kurt(dti_map, density_map)

    print(weightedMean, weightedStd, weightedSkew, weightedKurt)

def noddi(density_map, noddi_map):
    
    weightedMean = w_mean(noddi_map, density_map)
    weightedStd = np.sqrt(w_var(noddi_map, density_map))
    weightedSkew = w_skew(noddi_map, density_map)
    weightedKurt = w_kurt(noddi_map, density_map)

    print(weightedMean, weightedStd, weightedSkew, weightedKurt)

def diamond():
    pass

def mf():
    pass

metrics = {
    dti : ["FA", "AD", "RD", "MD"],
    noddi : ["icvf", "odi", "fbundle", "fextra", "fintra", "fiso" ], # (mu) Mean of the watson distribution of the Intra-cellular model is a 4D image with 2 values in the 4th
    #mf : ["fvf", "frac", "DIFF_ex"]
}


   
def tract_metrics(p_code, folder_path):
    subject_path = "%s/subjects/%s" % (folder_path, p_code)
    tracts_path = "%s/dMRI/tractography" % subject_path
    m = {}
    density_maps = {}

    for model, m_metrics in metrics.items():
        model_path = "%s/dMRI/microstructure/%s" % (subject_path, model.__name__)
        if not os.path.isdir(model_path):
            print("Model metrics don't exist")
            continue

        for metric in m_metrics:
            metric_path = None
            if model.__name__ == "dti":
                metric_path = "%s/%s_%s.nii.gz" % (model_path, p_code, metric)
            else:
                metric_path = "%s/%s_%s_%s.nii.gz" % (model_path, p_code, model.__name__, metric)

            metric_map = nib.load(metric_path).get_fdata()
            print("Extracting metric %s" % metric)
            for tract_filename in os.listdir(tracts_path):
                tract_name_ext = tract_filename.split(".")
                if len(tract_name_ext) != 2:
                    continue
                tract_name, ext = tract_name_ext
                if "fornix" not in tract_name : ## FOR TESTING
                    continue
                if ext == "trk":
                    tract_path = os.path.join(tracts_path, tract_filename)

                    # save in memory the density_map of the tract, in order to not open them every time, and speedup
                    density_map = None
                    if tract_path not in density_maps:
                        trk = load_tractogram(tract_path, "same")
                        trk.to_vox()
                        trk.to_corner()
                        density_map = get_streamline_density(trk)
                        density_maps[tract_path] = density_map
                    else:
                        density_map = density_maps[tract_path]

                    attr_name = "%s_%s" % (tract_name, metric)
                    print(attr_name)
                    m[attr_name] = model(density_map, metric_map)
                    print("")

tract_metrics("subj00", "../study")