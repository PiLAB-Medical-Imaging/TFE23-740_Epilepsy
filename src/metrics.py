import os
import sys
import pandas as pd
import numpy as np
import nibabel as nib
import subprocess
import json

from dipy.io.streamline import load_tractogram
from unravel.utils import *
from unravel.core import *
from nibabel import Nifti1Image
from numpy import linalg as LA
from scipy import stats

from params import *

def vcol(v):
    return v.reshape(v.size, 1)

def vrow(v):
    return v.reshape(1, v.size)

def pthPowerWeights(w, p):
    wp = np.power(w, p)
    return np.sum(wp)

def w_mean(v, w, p=1):
    assert v.size == w.size

    v = np.power(vcol(v), p)
    w = vrow(w)

    V1 = pthPowerWeights(w, 1)
    return np.ravel(np.dot(w,v)/V1)[0]

def w_mean_alt(v, w, K = 1000):
    v = v.ravel()
    w = w.ravel()
    w = w/np.sum(w) # normalization

    assert np.rint(w.sum()) == 1

    ra = np.random.random(v.size*K)

    #r = np.random.random(v.size*K)
    # si hanno gli stessi risultati ma è velocissimo, ma ovviamente prende troppo memoria e con K elevati si blocca tutto
    r = ra
    r = vcol(r)
    rep_col_r = np.tile(r, (1, w.size))

    cumsum = w.cumsum()
    cumsum = vrow(cumsum)
    rep_row_cumsum = np.tile(cumsum, (v.size*K, 1))

    weight_rand_idx = (rep_col_r <= rep_row_cumsum).argmax(axis=1)

    ##################

    #r = np.random.random(v.size*K)
    r = ra
    for i, rv in enumerate(r):
        r[i:i+1] = (rv <= w.cumsum()).argmax()
    r = np.array(r, dtype=int)
    

    print("\t", np.mean(v[r]))

    print("\t", np.mean(v[weight_rand_idx]))

    return 0

def w_var(v, w):
    v = vcol(v)
    w = vrow(w)

    v_centr = v - w_mean(v, w) # centerinig the values

    V_1 = pthPowerWeights(w, 1)
    V_2 = pthPowerWeights(w, 2)

    m_2 = w_mean(v_centr, w, p=2) 

    with np.errstate(divide='ignore', invalid='ignore'):
        M_2 = V_1**2/(V_1**2-V_2) * m_2 

    return M_2

def w_std_alt(v, w, K = 1000):
    v = v.ravel()
    w = w.ravel()
    w = w/np.sum(w) # normalization

    assert np.rint(w.sum()) == 1

    r = np.random.random(v.size*K)
    for i, rv in enumerate(r):
        r[i:i+1] = (rv <= w.cumsum()).argmax()
    r = np.array(r, dtype=int)

    return stats.tstd(v[r])

"""
@article{rimoldini2014weighted,
  title={Weighted skewness and kurtosis unbiased by sample size and Gaussian uncertainties},
  author={Rimoldini, Lorenzo},
  journal={Astronomy and Computing},
  volume={5},
  pages={1--8},
  year={2014},
  publisher={Elsevier}
}
"""
def w_skew(v, w):
    v = vcol(v)
    w = vrow(w)

    v_centr = v - w_mean(v, w) # centerinig the values

    V_1 = pthPowerWeights(w, 1)
    V_2 = pthPowerWeights(w, 2)
    V_3 = pthPowerWeights(w, 3)

    m_3 = w_mean(v_centr, w, p=3)

    with np.errstate(divide='ignore', invalid='ignore'):
        M_3 = (V_1**3)/(V_1**3 - 3*V_1*V_2 + 2*V_3) * m_3 

    return m_3

def w_skew(v, w, K = 1000):
    v = v.ravel()
    w = w.ravel()
    w = w/np.sum(w) # normalization

    assert np.rint(w.sum()) == 1

    r = np.random.random(v.size*K)
    for i, rv in enumerate(r):
        r[i:i+1] = (rv <= w.cumsum()).argmax()
    r = np.array(r, dtype=int)

    return stats.skew(v[r], bias=False)

"""
@article{rimoldini2014weighted,
  title={Weighted skewness and kurtosis unbiased by sample size and Gaussian uncertainties},
  author={Rimoldini, Lorenzo},
  journal={Astronomy and Computing},
  volume={5},
  pages={1--8},
  year={2014},
  publisher={Elsevier}
}
"""
def w_kurt(v, w):
    v = vcol(v)
    w = vrow(w)

    v_centr = v - w_mean(v, w) # centerinig the values

    V_1 = pthPowerWeights(w, 1)
    V_2 = pthPowerWeights(w, 2)
    V_3 = pthPowerWeights(w, 3)
    V_4 = pthPowerWeights(w, 4)

    m_2 = w_mean(v_centr, w, p=2)
    m_4 = w_mean(v_centr, w, p=4)

    denominator = (V_1**2 - V_2)*(V_1**4 - 6*V_1**2*V_2 + 8*V_1*V_3 + 3*V_2**2 - 6*V_4)
    with np.errstate(divide='ignore', invalid='ignore'):
        M_4 = V_1**2*(V_1**4 - 4*V_1*V_3 + 3*V_2**2)/denominator * m_4 - 3*V_1**2*(V_1**4 - 2*V_1**2*V_2 + 4*V_1*V_3 - 3*V_2**2)/denominator * m_2**2

    return M_4

def w_kurt(v, w, K = 1000):
    v = v.ravel()
    w = w.ravel()
    w = w/np.sum(w) # normalization

    assert np.rint(w.sum()) == 1

    r = np.random.random(v.size*K)
    for i, rv in enumerate(r):
        r[i:i+1] = (rv <= w.cumsum()).argmax()
    r = np.array(r, dtype=int)

    return stats.kurtosis(v[r], fisher=True, bias=False)

def MD(eigvals):
    return np.mean(eigvals, axis=3)

def FA(eigvals):
    md = MD(eigvals)
    md_expanded = np.repeat(md[:, :, :, np.newaxis], 3, axis=3) # only for centrering the values

    centred = eigvals - md_expanded
    num = np.sum(centred**2, axis=3)
    den = np.sum(eigvals**2, axis=3)
    res = np.sqrt(3/2 * np.divide(num, den))

    return np.nan_to_num(res)

def AxD(eigvals):
    # eigvals must be in ascending order (the last one is the biggest)
    return eigvals[:,:,:,-1]

def RD(eigvals):
    # eigvals must be in ascending order (the last one is the biggest)
    return np.mean(eigvals[:,:,:,:-1], axis=3)

def c_maps(t_img : Nifti1Image):
    t = t_img.get_fdata()
    t = t[:,:,:,0,:] # I don't know why there is a one more dimension in between

    tensors = np.zeros(t.shape[:3] + (3,3))
    comp_maps = np.zeros(t.shape[:3] + (4,))

    # Example for the upper part, we do only for the lower, because we use eigvalsh() with 'L'
    # i, j = [0, 0, 1], [1, 2, 2]
    # tensors[:,:,:,i,j] = t[:,:,:,[1,3,4]]
    i, j = np.tril_indices(3) # give back arrays of indices of a lower triangle matrix 3x3
    tensors[:,:,:,i,j] = t[:,:,:]

    eigs = LA.eigvalsh(tensors[:,:,:], 'L') # ascending ordered
    comp_maps[:,:,:,0] = FA(eigs)
    comp_maps[:,:,:,1] = MD(eigs)
    comp_maps[:,:,:,2] = AxD(eigs)
    comp_maps[:,:,:,3] = RD(eigs)

    return Nifti1Image(comp_maps, t_img.affine)

def save_DIAMOND_cMap_wMap_divideFract(diamond_fold, subj_id):
    t0_path = diamond_fold + "/" + subj_id + "_diamond_t0.nii.gz"
    t1_path = diamond_fold + "/" + subj_id + "_diamond_t1.nii.gz"
    fracs_path =  diamond_fold + "/" + subj_id + "_diamond_fractions.nii.gz"

    t0 : Nifti1Image = nib.load(t0_path)
    t1 : Nifti1Image = nib.load(t1_path)

    fracs : Nifti1Image = nib.load(fracs_path)
    frac_0 = nib.squeeze_image(fracs.slicer[..., 0]).get_fdata()
    frac_1 = nib.squeeze_image(fracs.slicer[..., 1]).get_fdata()
    frac_csf = nib.squeeze_image(fracs.slicer[..., 2]).get_fdata()

    weighted_maps = np.zeros(t0.shape[:3] + (4,), frac_0.dtype)
    c0_maps = c_maps(t0).get_fdata()
    c1_maps = c_maps(t1).get_fdata()

    # TODO here can be implemented the angular weighting through the UNRAVEL repo
    for i in range(4): # (FA, MD, AxD, RD)
        weighted_maps[:,:,:,i] = (frac_0*c0_maps[:,:,:,i] + frac_1*c1_maps[:,:,:,i]) / (frac_0+frac_1+frac_csf) # the denominator should be equal to 1 or close to 1 due to rounding (could be removed, but we leave it for clarity)
    weighted_maps = np.nan_to_num(weighted_maps)

    c0_img = Nifti1Image(c0_maps, t0.affine)
    c1_img = Nifti1Image(c1_maps, t1.affine)
    wFA_img = Nifti1Image(weighted_maps[:,:,:,0], t0.affine)
    wMD_img = Nifti1Image(weighted_maps[:,:,:,1], t0.affine)
    wAxD_img = Nifti1Image(weighted_maps[:,:,:,2], t0.affine)
    wRD_img = Nifti1Image(weighted_maps[:,:,:,3], t0.affine)

    # save the compartments maps in 4D [FA_c, MD_c, AxD_c, RD_c], just to check the results
    nib.save(c0_img, diamond_fold + "/" + subj_id + "_diamond_c0_DTI.nii.gz")
    nib.save(c1_img, diamond_fold + "/" + subj_id + "_diamond_c1_DTI.nii.gz")
    # save the weighted images
    nib.save(wFA_img, diamond_fold + "/" + subj_id + "_diamond_wFA.nii.gz")
    nib.save(wMD_img, diamond_fold + "/" + subj_id + "_diamond_wMD.nii.gz")
    nib.save(wAxD_img, diamond_fold + "/" + subj_id + "_diamond_wAxD.nii.gz")
    nib.save(wRD_img, diamond_fold + "/" + subj_id + "_diamond_wRD.nii.gz")
    # save the fraction.nii.gz file, but with fraction separated, one for each nifti image (not anymore in 4D)
    nib.save(nib.squeeze_image(fracs.slicer[..., 0]), diamond_fold + "/" + subj_id + "_diamond_frac_c0.nii.gz")
    nib.save(nib.squeeze_image(fracs.slicer[..., 1]), diamond_fold + "/" + subj_id + "_diamond_frac_c1.nii.gz")
    nib.save(nib.squeeze_image(fracs.slicer[..., 2]), diamond_fold + "/" + subj_id + "_diamond_frac_csf.nii.gz")

def trilinearInterpROI(subj_path, subj_id, masks : dict):
    # must exist the registration done in the tractography step
    registration_path = subj_path + "/registration"
    if not os.path.isdir(registration_path):
        return 1
    
    # get the ids of the regions
    rois = []
    for nums in masks.values():
        for num in nums:
            rois.append(num)
    
    # get the names of the regions
    # TODO Basically this is the same code as tractography.py, all can be organized in a different way to not have code duplication
    rois.sort()
    k = 0
    roi_num_name = {}
    colorLUT_path = os.getenv('FREESURFER_HOME') + "/FreeSurferColorLUT.txt"
    with open(colorLUT_path, "r") as f:
        for line in f.readlines():
            elems = line.split()
            if len(elems) == 0 or not elems[0].isdigit() :
                continue
            curr_num = int(elems[0])
            curr_name = elems[1]
            if curr_num == rois[k]:
                roi_num_name[curr_num] = curr_name
                k += 1
            if k == len(rois):
                break

    # extraction of the ROI NOT registered, then binarize them, and register them with a Trilinear Interpolation
    masksNotReg_path = "%s/masks/notReg" % subj_path
    if not os.path.isdir(masksNotReg_path):
        os.mkdir(masksNotReg_path)
    trilInterp_paths = []
    for num, name in roi_num_name.items():
        out_path = "%s/%s_%s_aparc+aseg-NoReg.nii.gz" % (masksNotReg_path, subj_id, name)
        cmd = "mri_extract_label -exit_none_found %s/mri/aparc+aseg.mgz %d %s" % (subj_path, num, out_path)

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        process.wait()
        if process.returncode != 0:
            return 1

        # binarize the roi, because the freesurfer roi are not {0, 1}, but with a label
        mask_map : Nifti1Image = nib.load(out_path)
        mask_np = mask_map.get_fdata()
        mask_np[mask_np > 0] = 1 # binarizaton
        mask_map = Nifti1Image(mask_np, mask_map.affine)
        nib.save(mask_map, out_path) # we overwrite the old one

        # Trilinear interpolation
        print("Interpolation: %s" % (name))
        out_trilInterp_path = "%s/masks/%s_%s-TrilInterp_aparc+aseg.nii.gz" % (subj_path, subj_id, name)
        cmd = "mri_vol2vol --reg %s/transf_dMRI_t1.dat --targ %s --mov %s/dMRI/preproc/%s_dmri_preproc.nii.gz --o %s --interp trilin --inv" % (registration_path, out_path, subj_path, subj_id, out_trilInterp_path)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        process.wait()
        if process.returncode != 0:
            print("Error freesurfer mri_vol2vol aseg")
            return 1
        
        trilInterp_paths.append((name, out_trilInterp_path))
    return trilInterp_paths

"""
Explain of the correction in the thesis
"""
def correctWeightsTract(weights, nStreamLines):
    n_fascs, n_voxs = np.unique(weights, return_counts=True)
    for n_fasc, n_vox in zip(n_fascs, n_voxs):
        weights[weights == n_fasc] = weights[weights == n_fasc] / n_vox
    #return weights/nStreamLines
    return weights/np.sum(weights)

metrics = {
    "dti" : ["FA", "AD", "RD", "MD"],
    "noddi" : ["icvf", "odi", "fbundle", "fextra", "fintra", "fiso" ],
    "diamond" : ["wFA", "wMD", "wAxD", "wRD", "frac_c0", "frac_c1", "frac_csf"],
    "mf" : ["fvf_f0", "fvf_f1", "fvf_tot", "frac_f0", "frac_f1", "frac_csf"]
}

# Freesurfer LUT: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
masks_freesurfer = {
    "thalamus" : [10, 49],
    "hippocampus" : [17, 53],
    "amygdala" : [18, 54],
    "accumbens" : [26, 58],
    "putamen" : [12, 51],
    "pallidum" : [13, 52],
}

def compute_metricsPerROI(p_code, folder_path):
    subject_path = "%s/subjects/%s" % (folder_path, p_code)
    tracts_path = "%s/dMRI/tractography" % subject_path
    m = {}
    m["ID"] = p_code
    density_maps = {}

    def addMetrics(roi_name, metric, model, metric_map, density_map):
        attr_name = "%s_%s" % (roi_name, metric)
        if "frac_csf" == metric:
            if "diamond" == model:
                attr_name += "_d"
            elif "mf" == model:
                attr_name += "_mf"

        m[attr_name + "_mean"] = w_mean(metric_map, density_map)
        m[attr_name + "_std"] = np.sqrt(w_var(metric_map, density_map))
        m[attr_name + "_skew"] = w_skew(metric_map, density_map)
        m[attr_name + "_kurt"] = w_kurt(metric_map, density_map)

    # Trilinear interpolation of the segments into dMRI space
    # We consider the interpolated voxels as a weighted mask (density mask)
    trilInterp_paths = trilinearInterpROI(folder_path+"/subjects/"+p_code, p_code, masks_freesurfer)
    if(trilInterp_paths == 1):
        print("Error during te trilinear interpolation")
        return 1

    for model, m_metrics in metrics.items():
        model_path = "%s/dMRI/microstructure/%s/" % (subject_path, model)
        if not os.path.isdir(model_path):
            print("Model metrics don't exist")
            continue

        if model == "diamond":
            save_DIAMOND_cMap_wMap_divideFract(model_path, p_code) # Compute wFA, wMD, wAxD, wRD and save it in nifti file

        for metric in m_metrics:
            metric_path = None
            if model == "dti":
                metric_path = "%s/%s_%s.nii.gz" % (model_path, p_code, metric)
            elif model in ["noddi", "diamond", "mf"]:
                metric_path = "%s/%s_%s_%s.nii.gz" % (model_path, p_code, model, metric)
            
            metric_map : Nifti1Image = nib.load(metric_path)
            affine_info = metric_map.affine
            metric_map = metric_map.get_fdata()

            for tract_filename in os.listdir(tracts_path):
                tract_name_ext = tract_filename.split(".")
                if len(tract_name_ext) != 2:
                    continue
                tract_name, ext = tract_name_ext

                if "rmvd" in tract_name:
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
                        nib.save(nib.Nifti1Image(density_map, affine_info), "%s/masks/%s_%s_tractNoCorr.nii.gz" % (subject_path, p_code, tract_name))

                        m[tract_name + "_nTracts"] =  get_streamline_count(trk)
                        density_map = correctWeightsTract(density_map, m[tract_name + "_nTracts"])
                        
                        density_maps[tract_path] = density_map
                        nib.save(nib.Nifti1Image(density_map, affine_info), "%s/masks/%s_%s_tract.nii.gz" % (subject_path, p_code, tract_name))
                    else:
                        density_map = density_maps[tract_path]

                    addMetrics(tract_name, metric, model, metric_map, density_map)
                        

            for mask_name, mask_path in trilInterp_paths:
                # save in memory the density_map of the mask, in order to not open them every time, and speedup
                density_map = None
                if mask_path not in density_maps:
                    density_map = nib.load(mask_path).get_fdata()
                    density_maps[mask_path] = density_map
                else:
                    density_map = density_maps[mask_path]

                addMetrics(mask_name.lower(), metric, model, metric_map, density_map)

    print(json.dumps(m,indent=2, sort_keys=True))
    with open("%s/dMRI/microstructure/%s_metrics.json" % (subject_path, p_code), "w") as outfile:
        json.dump(m, outfile, indent=2, sort_keys=True)
                
    df = pd.DataFrame([m])
    df.to_csv("%s/dMRI/microstructure/%s_metrics.csv" % (subject_path, p_code), index=False)

def main():
    
    ## Getting folder
    folder_path = get_folder(sys.argv)

    ## Get the patient
    p = get_patient(sys.argv)

    compute_metricsPerROI(p, folder_path)

if __name__ == "__main__":
    exit(main())