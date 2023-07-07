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
from statsmodels.stats.weightstats import DescrStatsW

from params import *

def vcol(v):
    return v.reshape(v.size, 1)

def vrow(v):
    return v.reshape(1, v.size)

def pthPowerWeights(w, p):
    wp = np.power(w, p)
    return np.sum(wp)

def w_mean(v, w, p=1):
    assert v.shape == w.shape or v.size == w.size

    v = np.power(vcol(v), p)
    w = vrow(w)

    V1 = pthPowerWeights(w, 1)
    return np.ravel(np.dot(w,v)/V1)[0]

def w_mean_alt(v, w, K = 1000):
    assert v.shape == w.shape or v.size == w.size
    v = v[w>0]
    w = w[w>0]
    w = w/np.sum(w) # normalization

    assert np.rint(w.sum()) == 1

    ra = np.random.random(v.size*K)

    ## #r = np.random.random(v.size*K)
    ## # si hanno gli stessi risultati ma è velocissimo, ma ovviamente prende troppo memoria e con K elevati si blocca tutto
    ## r = ra
    ## r = vcol(r)
    ## rep_col_r = np.tile(r, (1, w.size))
## 
    ## cumsum = w.cumsum()
    ## cumsum = vrow(cumsum)
    ## rep_row_cumsum = np.tile(cumsum, (v.size*K, 1))
## 
    ## weight_rand_idx = (rep_col_r <= rep_row_cumsum).argmax(axis=1)
## 
    ## ##################

    #r = np.random.random(v.size*K)
    r = ra
    for i, rv in enumerate(r):
        r[i:i+1] = (rv <= w.cumsum()).argmax()
    r = np.array(r, dtype=int)

    # print("\t", np.mean(v[weight_rand_idx]))

    return np.mean(v[r])

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
def w_var(v, w):
    assert v.shape == w.shape
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
    assert v.shape == w.shape
    v = v[w>0].ravel()
    w = w[w>0].ravel()
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
    assert v.shape == w.shape
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
    assert v.shape == w.shape
    v = v[w>0].ravel()
    w = w[w>0].ravel()
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
    assert v.shape == w.shape
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
    assert v.shape == w.shape
    v = v[w>0].ravel()
    w = w[w>0].ravel()
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
    res = np.sqrt(3/2 * np.divide(num, den, out=np.zeros_like(num), where=den!=0))

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
    den = frac_0+frac_1
    for i in range(4): # (FA, MD, AxD, RD)
        num = (frac_0*c0_maps[:,:,:,i] + frac_1*c1_maps[:,:,:,i])
        weighted_maps[:,:,:,i] = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
    weighted_maps = np.nan_to_num(weighted_maps)

    c0_img = Nifti1Image(c0_maps, t0.affine)
    c1_img = Nifti1Image(c1_maps, t1.affine)
    wFA_img = Nifti1Image(weighted_maps[:,:,:,0], t0.affine)
    wMD_img = Nifti1Image(weighted_maps[:,:,:,1], t0.affine)
    wAxD_img = Nifti1Image(weighted_maps[:,:,:,2], t0.affine)
    wRD_img = Nifti1Image(weighted_maps[:,:,:,3], t0.affine)
    frac_ftot_img = Nifti1Image(den, t0.affine)

    # save the compartments maps in 4D [FA_c, MD_c, AD_c, RD_c], just to check the results
    nib.save(c0_img, diamond_fold + "/" + subj_id + "_diamond_c0_DTI.nii.gz")
    nib.save(c1_img, diamond_fold + "/" + subj_id + "_diamond_c1_DTI.nii.gz")
    # save the weighted images
    nib.save(wFA_img, diamond_fold + "/" + subj_id + "_diamond_wFA.nii.gz")
    nib.save(wMD_img, diamond_fold + "/" + subj_id + "_diamond_wMD.nii.gz")
    nib.save(wAxD_img, diamond_fold + "/" + subj_id + "_diamond_wAD.nii.gz")
    nib.save(wRD_img, diamond_fold + "/" + subj_id + "_diamond_wRD.nii.gz")
    # save the fraction.nii.gz file, but with fraction separated, one for each nifti image (not anymore in 4D)
    nib.save(nib.squeeze_image(fracs.slicer[..., 0]), diamond_fold + "/" + subj_id + "_diamond_frac_c0.nii.gz")
    nib.save(nib.squeeze_image(fracs.slicer[..., 1]), diamond_fold + "/" + subj_id + "_diamond_frac_c1.nii.gz")
    nib.save(nib.squeeze_image(fracs.slicer[..., 2]), diamond_fold + "/" + subj_id + "_diamond_frac_csf.nii.gz")
    nib.save(frac_ftot_img, diamond_fold + "/" + subj_id + "_diamond_frac_ctot.nii.gz")

def save_mf_wfvf(mf_fold, subj_id):
    frac_0_path = mf_fold + "/" + subj_id + "_mf_frac_f0.nii.gz"
    frac_1_path = mf_fold + "/" + subj_id + "_mf_frac_f1.nii.gz"
    frac_csf_path = mf_fold + "/" + subj_id + "_mf_frac_csf.nii.gz"
    fvf_f0_path = mf_fold + "/" + subj_id + "_mf_fvf_f0.nii.gz"
    fvf_f1_path = mf_fold + "/" + subj_id + "_mf_fvf_f1.nii.gz"

    frac_0_map : Nifti1Image = nib.load(frac_0_path)
    frac_1_map : Nifti1Image = nib.load(frac_1_path)
    frac_csf_map : Nifti1Image = nib.load(frac_csf_path)
    fvf_0_map : Nifti1Image = nib.load(fvf_f0_path)
    fvf_1_map : Nifti1Image = nib.load(fvf_f1_path)

    frac_0 = frac_0_map.get_fdata()
    frac_1 = frac_1_map.get_fdata()
    frac_csf = frac_csf_map.get_fdata()
    fvf_0 = fvf_0_map.get_fdata()
    fvf_1 = fvf_1_map.get_fdata()

    num = (frac_0*fvf_0 + frac_1*fvf_1)
    den = (frac_0 + frac_1)
    wfvf = np.divide(num, den, out=np.zeros_like(num), where=den!=0)

    wfvf_map = Nifti1Image(wfvf, fvf_0_map.affine)
    frac_ftot_map = Nifti1Image(den, fvf_0_map.affine)

    nib.save(wfvf_map, mf_fold + "/" + subj_id + "_mf_wfvf.nii.gz")
    nib.save(frac_ftot_map, mf_fold + "/" + subj_id + "_mf_frac_ftot.nii.gz")

def trilinearInterpROI(folder_path, subj_id, masks : dict):
    # must exist the registration done in the tractography step
    subj_path = "%s/subjects/%s" % (folder_path, subj_id)
    freesurfer_path = "%s/freesurfer/%s" % (folder_path, subj_id)

    registration_path = subj_path + "/registration"
    if not os.path.isdir(registration_path):
        raise Exception
    
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
        cmd = "mri_extract_label -exit_none_found %s/mri/aparc+aseg.mgz %d %s" % (freesurfer_path, num, out_path)

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        process.wait()
        if process.returncode != 0:
            raise Exception

        # binarize the roi, because the freesurfer roi are not {0, 1}, but with a label
        mask_map : Nifti1Image = nib.load(out_path)
        mask_np = mask_map.get_fdata()
        mask_np[mask_np > 0] = 1 # binarizaton
        volume = mask_np.sum() # the volume of the roi is compute before the tranformation to not loss voxels after the registration
        mask_map = Nifti1Image(mask_np, mask_map.affine)
        nib.save(mask_map, out_path) # we overwrite the old one

        # Trilinear interpolation
        # print("Interpolation: %s" % (name))
        out_trilInterp_path = "%s/masks/%s_%s-TrilInterp_aparc+aseg.nii.gz" % (subj_path, subj_id, name)
        cmd = "mri_vol2vol --reg %s/transf_dMRI_t1.dat --targ %s --mov %s/dMRI/preproc/%s_dmri_preproc.nii.gz --o %s --interp trilin --inv" % (registration_path, out_path, subj_path, subj_id, out_trilInterp_path)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        process.wait()
        if process.returncode != 0:
            print("Error freesurfer mri_vol2vol aseg")
            raise Exception
        
        trilInterp_paths.append((name, out_trilInterp_path, volume))
    return trilInterp_paths

def decresingSigmoid(x, min=0.1, max=0.5, x_max=20):
    """
    Every unit of the smothing the funcion si dilated by 5 from a side
    """
    def sig(x):
        return 1/(1+np.exp(-x))
    
    flex = x_max/2
    smooth = flex/5

    if x > x_max:
        return min
    return sig(-(x-flex)/smooth)*(max-min)+min 

"""
Explain of the correction in the thesis
"""
# def correctWeightsTract(weights, nTracts):
#     # minmax scaler in [0, 1]
#     weights_scaled = (weights - weights.min()) / (weights.max() - weights.min())
#     # threshold
#     thresh = decresingSigmoid(nTracts)
#     weights[weights_scaled<thresh] = 0
#     # give more weight to voxels with more weight and less to others
#     # weights = weights * np.exp(weights) 
#     
#     return weights

def correctWeightsTract(weights, thresh=0.45):
    from scipy import signal
    kernel = [[[1/2, 1/2, 1/2],
               [1/2, 1/2, 1/2],
               [1/2, 1/2, 1/2]],
              [[1/2, 1/2, 1/2],
               [1/2, 1  , 1/2],
               [1/2, 1/2, 1/2]],
              [[1/2, 1/2, 1/2],
               [1/2, 1/2, 1/2],
               [1/2, 1/2, 1/2]]]
    kernel = np.array(kernel)

    c = signal.convolve(weights, kernel, mode="same", method="direct")
    c_scal = (c - c.min())/(c.max() - c.min())
    c[c_scal<thresh] = 0
    c[weights == 0] = 0 # because during the convolution are chosen also voxels that are not from the bundle, so we keep only the streamlines of the tract

    return c

# def mostImportant(weights):
#     # minmax scaler in [0, 1]
#     weights = (weights - weights.min()) / (weights.max() - weights.min())
#     # threshold
#     weights[weights<0.5] = 0
#     return weights

metrics = {
    "dti" : ["FA", "AD", "RD", "MD"],
    "noddi" : ["icvf", "odi", "fbundle", "fextra", "fintra", "fiso" ],
    "diamond" : ["wFA", "wMD", "wAD", "wRD", "frac_c0", "frac_c1", "frac_csf", "frac_ctot"],
    "mf" : ["fvf_f0", "fvf_f1", "wfvf", "fvf_tot", "frac_f0", "frac_f1", "frac_csf", "frac_ftot"]
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
    freesurfer_path = "%s/freesurfer/%s" % (folder_path, p_code)
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

        assert metric_map.shape == density_map.shape

        v = metric_map.ravel()
        w = density_map.ravel()

        assert v.size == w.size

        if w.sum() == 0:
            print(attr_name, "completed")
            return

        dstat = DescrStatsW(v, w)

        w_discrete = np.round(w).astype(int)
        repeat = np.repeat(v, w_discrete)

        m[attr_name + "_mean"] = np.average(v, weights=w)
        m[attr_name + "_std"] = dstat.std
        #m[attr_name + "_skew"] = w_skew(metric_map, density_map)
        m[attr_name + "_skew"] = stats.skew(repeat, bias=False)
        #m[attr_name + "_kurt"] = w_kurt(metric_map, density_map)
        m[attr_name + "_kurt"] = stats.kurtosis(repeat, fisher=True, bias=False)
        m[attr_name + "_max"] = metric_map[density_map>0].max()
        m[attr_name + "_min"] = metric_map[density_map>0].min()
        
        assert m[attr_name + "_min"] <= m[attr_name + "_mean"] and m[attr_name + "_mean"] <= m[attr_name + "_max"]

        print(attr_name, "completed")

    # Trilinear interpolation of the segments into dMRI space
    # We consider the interpolated voxels as a weighted mask (density mask)
    trilInterp_paths = trilinearInterpROI(folder_path, p_code, masks_freesurfer)

    for model, m_metrics in metrics.items():
        model_path = "%s/dMRI/microstructure/%s/" % (subject_path, model)
        if not os.path.isdir(model_path):
            print("Model metrics don't exist")
            continue

        # frac_fasci_path = None
        # frac_fasci_mask = None

        if model == "diamond":
            save_DIAMOND_cMap_wMap_divideFract(model_path, p_code) # Compute wFA, wMD, wAxD, wRD and save it in nifti file
            # frac_fasci_path = "%s/%s_%s_frac_ctot.nii.gz" % (model_path, p_code, model)

        if model == "mf":
            save_mf_wfvf(model_path, p_code) # compute mf_wfvf
            # frac_fasci_path = "%s/%s_%s_frac_ftot.nii.gz" % (model_path, p_code, model)

        # if model in ["diamond", "mf"]:
        #     frac_fasci_mask = nib.load(frac_fasci_path).get_fdata()
        #     frac_fasci_mask[frac_fasci_mask > 0] = 1

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
            
                        if not os.path.isfile("%s/masks/%s_%s_tractNoCorr.nii.gz" % (subject_path, p_code, tract_name)):
                            # get the density
                            density_map = get_streamline_density(trk)
                            # save the density
                            bin_density_map = density_map.copy()
                            bin_density_map[bin_density_map > 0] = 1 # for visualization reasons
                            nib.save(nib.Nifti1Image(density_map, affine_info), "%s/masks/%s_%s_tractNoCorr.nii.gz" % (subject_path, p_code, tract_name))
                            nib.save(nib.Nifti1Image(bin_density_map, affine_info), "%s/masks/%s_%s_tractNoCorrBin.nii.gz" % (subject_path, p_code, tract_name))
                        else :
                            # load the density
                            density_map = nib.load("%s/masks/%s_%s_tractNoCorr.nii.gz" % (subject_path, p_code, tract_name)).get_fdata()
                        # add as feaure the number of tracts of the tract
                        m[tract_name + "_nTracts"] =  get_streamline_count(trk)
            
                        density_map = correctWeightsTract(density_map)
                        
                        # save in memory to be faster
                        density_maps[tract_path] = density_map
                        # save the corrected density
                        bin_density_map = density_map.copy()
                        bin_density_map[bin_density_map > 0] = 1 # for visualization reasons
                        nib.save(nib.Nifti1Image(density_map, affine_info), "%s/masks/%s_%s_%tract.nii.gz" % (subject_path, p_code, tract_name))
                        nib.save(nib.Nifti1Image(bin_density_map, affine_info), "%s/masks/%s_%s_tractBin.nii.gz" % (subject_path, p_code, tract_name))
                    else:
                        # Use the density map saved in memory
                        density_map = density_maps[tract_path]
            
                    addMetrics(tract_name, metric, model, metric_map, density_map)
                        

            for mask_name, mask_path, volume in trilInterp_paths:
                # save in memory the density_map of the mask, in order to not open them every time, and speedup
                density_map = None
                if mask_path not in density_maps:
                    density_map = nib.load(mask_path).get_fdata()
                    density_maps[mask_path] = density_map
                else:
                    density_map = density_maps[mask_path]
            
                m[mask_name.lower() + "_voxVol"] = volume
            
                addMetrics(mask_name.lower(), metric, model, metric_map, density_map)

            for (dir_path, _, file_names) in os.walk(f"{freesurfer_path}/dpath"):
                for file_name in file_names:
                    if file_name.endswith(".map.nii.gz"):
                        mask_path = os.path.join(dir_path,file_name)
                        mask_name = dir_path.split("/")[-1].split("_")[0]

                        density_map = None
                        if mask_path not in density_maps:
                            density_map = nib.load(mask_path).get_fdata()
                            density_maps[mask_path] = density_map
                        else:
                            density_map = density_maps[mask_path]
                        
                        addMetrics(mask_name.lower(), metric, model, metric_map, density_map)

    ## print(json.dumps(m,indent=2, sort_keys=True))
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