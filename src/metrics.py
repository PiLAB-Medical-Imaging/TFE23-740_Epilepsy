import os
import sys
import pandas as pd
import numpy as np
import nibabel as nib
import subprocess
import json
import SimpleITK as sitk

from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram, Origin
from unravel.utils import *
from unravel.core import *
from unravel.stream import smooth_streamlines
from nibabel import Nifti1Image
from numpy import linalg as LA
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW
from radiomics import featureextractor
from tqdm import tqdm

from params import *

def vcol(v):
    return v.reshape(v.size, 1)

def vrow(v):
    return v.reshape(1, v.size)

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
    if (os.path.isfile(diamond_fold + "/" + subj_id + "_diamond_c0_DTI.nii.gz") and
        os.path.isfile(diamond_fold + "/" + subj_id + "_diamond_c1_DTI.nii.gz") and
        os.path.isfile(diamond_fold + "/" + subj_id + "_diamond_wFA.nii.gz") and
        os.path.isfile(diamond_fold + "/" + subj_id + "_diamond_wMD.nii.gz") and
        os.path.isfile(diamond_fold + "/" + subj_id + "_diamond_wRD.nii.gz") and
        os.path.isfile(diamond_fold + "/" + subj_id + "_diamond_wAD.nii.gz") and
        os.path.isfile(diamond_fold + "/" + subj_id + "_diamond_frac_c0.nii.gz") and
        os.path.isfile(diamond_fold + "/" + subj_id + "_diamond_frac_c1.nii.gz") and
        os.path.isfile(diamond_fold + "/" + subj_id + "_diamond_frac_csf.nii.gz") and
        os.path.isfile(diamond_fold + "/" + subj_id + "_diamond_frac_ctot.nii.gz")):
        return

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
    if (os.path.isfile(mf_fold + "/" + subj_id + "_mf_wfvf.nii.gz") and
        os.path.isfile(mf_fold + "/" + subj_id + "_mf_frac_ftot.nii.gz")):
        return

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

def correctWeightsTract(weights, thresh=0.1):
    def resize_and_fix_origin(kernel, size):
        """Pads a kernel to reach shape `size`, and shift it in order to cancel phase.
        This is based on the assumption that the kernel is centered in image space.
        A note about this is provided in section 1.4. 
        """
        # Very specific routine... You don't have to understand it
        pad0, pad1, pad2 = size[0]-kernel.shape[0], size[1]-kernel.shape[1], size[2]-kernel.shape[2]
        # shift less if kernel is even, to start with 2 central items
        shift0, shift1, shift2 = (kernel.shape[0]-1)//2, (kernel.shape[1]-1)//2, (kernel.shape[2]-1)//2

        kernel = np.pad(kernel, ((0,pad0), (0,pad1), (0,pad2)), mode='constant')
        kernel = np.roll(kernel, (-shift0, -shift1, -shift2), axis=(0,1,2))
        return kernel

    from scipy import signal
    kernel = [[[1/4, 1/2, 1/4],
               [1/2, 1, 1/2],
               [1/4, 1/2, 1/4]],
              [[1/2, 1, 1/2],
               [1,   4, 1],
               [1/2, 1, 1/2]],
              [[1/4, 1/2, 1/4],
               [1/2, 1, 1/2],
               [1/4, 1/2, 1/4]]]
    kernel = np.array(kernel)

    ftImage = np.fft.fftn(weights)
    ftKernel = np.fft.fftn(resize_and_fix_origin(kernel, weights.shape))
    c = np.real(np.fft.ifftn(ftImage * ftKernel))
    # c = signal.convolve(weights, kernel, mode="same", method="direct")

    c_scal = (c - c.min())/(c.max() - c.min())
    c[c_scal<thresh] = 0
    c[weights == 0] = 0 # because during the convolution are chosen also voxels that are not from the bundle, so we keep only the streamlines of the tract

    return c

def highestProbTractsDensity(trk, wm: None):
    '''
    This function is a modified version of get_streamline_density() from the library RAVEL by Delinte Nicolas.
    The function is also a semplified version, it doesn't take the paramiters for the resolution_increase and the color.
    
    Paramiters
    ----------
    trk : tractogram
        Content of a .trk file

    Returns
    -------

    '''
    density = get_streamline_density(trk, subsegment=1) # The density
    isWM = wm > 0 
    notWM = wm == 0

    # Find the density for each streamline
    sList = tract_to_streamlines(trk)
    totDensityTract = np.zeros(len(sList))

    for j, streamline in enumerate(sList):
        # Create a new tract with a single streamline
        temp_trk = StatefulTractogram.from_sft([streamline], trk)
        # Get the ROI of the streamline
        roi = get_streamline_density(temp_trk, subsegment=1)

        # Se WM mask esiste => Solo i tratti che hanno una probabilità maggiore nella wm vengono considerati

        if wm is None:
            temp_lenght = np.sum(roi) # Normalizzo per la lunghezza del tratto, sennò i tratti più lunghi sarebbero avvantaggiati rispetto a quelli corti.
        else:
            temp_lenght = np.sum(roi[isWM]) # Correggo la lunghezza escludendo i voxel che vengono esclusi dalla regione, perche non appartenenti alla WM
            roi[notWM] = 0 # rimuovo voxels non appartenenti alla wm

        # Compute the density of the streamline
        totDensityTract[j] = np.sum(density[roi>0])/temp_lenght

    bestTracts_idx = np.argsort(totDensityTract)[::-1][0:int(0.01*get_streamline_count(trk))]

    streamlines = []
    for i in bestTracts_idx:
        streamlines.append(sList[i])

    temp_trk = StatefulTractogram.from_sft(streamlines, trk)
    roi = get_streamline_density(temp_trk, subsegment=1)
    density[roi==0] = 0
    if wm is not None:
        density[notWM] = 0

    return density

def getDictionaryFromLUT(lut_path):
    d = {}
    with open(lut_path, 'r') as file:
        for line in file.readlines():
            fields = line.split()
            if len(fields) == 0 or fields[0].startswith("#"):
                continue
            d[int(fields[0])] = fields[1]
    return d

metrics = {
    "dti" : ["FA" , "AD", "RD", "MD"],
    "noddi" : ["icvf", "odi", "fextra", "fiso"],
    "diamond" : ["wFA", "wMD", "wAD", "wRD", "frac_csf"],
    "mf" : ["wfvf", "fvf_tot", "frac_csf"]
}

# Freesurfer LUT: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
masks_freesurfer = {
    "Thalamus_union" : [(8103, 8134), (8203, 8234)],
    "hippocampus" : [17, 53],
    "amygdala" : [18, 54],
    "accumbens" : [26, 58],
    "putamen" : [12, 51],
    "pallidum" : [13, 52],
    "caudate" : [11, 50],
    "vessel" : [30, 62]
}

def compute_metricsPerROI(p_code, folder_path):
    # Folder paths
    subject_path = "%s/subjects/%s" % (folder_path, p_code)
    freesurfer_path = "%s/freesurfer/%s" % (folder_path, p_code)
    params = folder_path + "/static_files/radiomics_params.yaml"

    # metrics dictionary
    m = {}
    m["ID"] = p_code

    # Segmented brain
    seg_path = f"{freesurfer_path}/dlabel/diff/aparc+aseg+thalnuc.bbr.nii.gz"
    seg_map = nib.load(seg_path)
    seg = seg_map.get_fdata()

    # White Matter mask for tracts
    wm = nib.load(f"{freesurfer_path}/dlabel/diff/White-Matter++.bbr.nii.gz").get_fdata()
    wm[(seg>=251) & (seg<=255)] = 1 # We are including the Corpus Callosum in the WM

    # density maps saved in memory to save time
    idx_ROIpaths = []
    densities = []

    # create folder for tracts if it doesn't exist
    tract_fold = os.path.join(subject_path,"masks","tract")
    if not os.path.exists(tract_fold):
        os.mkdir(os.path.join(subject_path,"masks","tract"))

    # get dictionary from the LUT
    colorLUT = os.getenv('FREESURFER_HOME') + "/FreeSurferColorLUT.txt"
    dict_idx_ROI = getDictionaryFromLUT(colorLUT)

    dpath = f"{freesurfer_path}/dpath"
    for entry in tqdm(os.listdir(dpath)):
        entry_path = os.path.join(dpath, entry)
        if os.path.isdir(entry_path):
            trk_name = entry.split("_")[0]
            trk_path = os.path.join(entry_path, "path.pd.trk")
    
            # # DEBUG
            # if "acomm" not in trk_name:
            #     continue
            
            density_out_path = os.path.join(subject_path,"masks","tract",f"{p_code}_{trk_name}_tract.nii.gz")
            temp_trk_path = os.path.join(subject_path,"dMRI","tractography", f"{trk_name}_temp.trk")
    
            # Smoothing is needed since the freesurfer tracts aren't 
            smooth_streamlines(trk_path, temp_trk_path)
            smooth_streamlines(temp_trk_path, temp_trk_path)
    
            trk = load_tractogram(temp_trk_path, "same")
            trk.to_vox()
            trk.to_corner()
    
            # Get the density filtered by the higest prob streamlines
            idx_ROIpaths.append(density_out_path)
            density = correctWeightsTract(highestProbTractsDensity(trk, wm))
            density = np.where(density>0, 1, 0).astype("int32")
            densities.append(density)
    
            nib.save(nib.Nifti1Image(density, trk.affine), density_out_path)
            os.remove(temp_trk_path)

    # ------- TO COMPUTE THE METRICS ON THE COMPUTED TRACTS BY MRTRIX3, UNCOMMENT --------
    # tract_path = f"{subject_path}/dMRI/tractography"
    # for entry in os.listdir(tract_path):
    #     trk_path = os.path.join(tract_path, entry)
    #     if( os.path.isfile(trk_path) and 
    #         trk_path.endswith(".trk") and 
    #         not "rmvd" in trk_path ):
    # 
    #         trk_name = entry.split(".")[0]
    # 
    #         # DEBUG
    #         if "left-fornix" not in entry:
    #             continue
    # 
    #         density_out_path = os.path.join(subject_path,"masks","tract",f"{p_code}_{trk_name}_tract.nii.gz")
    # 
    #         trk = load_tractogram(trk_path, "same")
    #         trk.to_vox()
    #         trk.to_corner()
    # 
    #         print(trk_name)
    # 
    #         # Get the binary density filtered by the higest prob streamlines and with corrected weights
    #         idx_ROIpaths.append(density_out_path)
    #         density = correctWeightsTract(highestProbTractsDensity(trk, wm))
    #         density = np.where(density>0, 1, 0).astype("int32")
    #         densities.append(density)
    # 
    #         nib.save(nib.Nifti1Image(density, trk.affine), density_out_path)

    for name, roi_idxs in masks_freesurfer.items():

        for i, roi_idx in enumerate(roi_idxs):
            start, end = -1, -1
            if "union" in name:
                start, end = roi_idx

                roi_name = name.split("_")[0]
                roi_name = f"Left-{roi_name}" if i == 0 else f"Right-{roi_name}"
            else:
                roi_name = dict_idx_ROI[roi_idx]

            # # DEBUG
            # if roi_idx != 18:
            #     continue

            density_out_path = os.path.join(subject_path,"masks","tract",f"{p_code}_{roi_name}_diff.nii.gz")
            
            idx_ROIpaths.append(density_out_path)
            if "union" in name:
                density = np.where(((seg>=start) & (seg<=end)), 1, 0).astype("int32")
            else:
                density = np.where(seg == roi_idx, 1, 0).astype("int32")
            densities.append(density)

            nib.save(nib.Nifti1Image(density, seg_map.affine), density_out_path)
    
    def addMetrics(roi_name, metric, model, metric_path, roi_path):
        attr_name = "%s_%s" % (roi_name, metric)
        if "frac_csf" == metric:
            if "diamond" == model:
                attr_name += "_d"
            elif "mf" == model:
                attr_name += "_mf"

        print(attr_name, "Started")

        metric = sitk.ReadImage(metric_path)
        roi = sitk.ReadImage(roi_path)

        # I didnt get this passage, but it works
        metric = sitk.GetImageFromArray(sitk.GetArrayFromImage(metric))
        roi = sitk.GetImageFromArray(sitk.GetArrayFromImage(roi))

        extractor = featureextractor.RadiomicsFeatureExtractor(params)
        extractor.enableAllImageTypes()
        extractor.enableAllFeatures()
        results = extractor.execute(metric, roi)

        # Remove the diagnostic data, and leave only the features
        df_names = pd.DataFrame(columns=results.keys())
        diagnostics = df_names.filter(regex=r'diagnostics').columns
        features = df_names.drop(diagnostics, axis=1).columns

        for feature, value in results.items():
            if feature in features:
                try:
                    m[attr_name+"_"+feature] = value.item()
                except:
                    m[attr_name+"_"+feature] = value

        print(attr_name, "completed")

    for model, m_metrics in metrics.items():
        model_path = "%s/dMRI/microstructure/%s/" % (subject_path, model)
        if not os.path.isdir(model_path):
            print("Model metrics don't exist")
            continue

        if model == "diamond":
            save_DIAMOND_cMap_wMap_divideFract(model_path, p_code) # Compute wFA, wMD, wAxD, wRD and save it in nifti file

        if model == "mf":
            save_mf_wfvf(model_path, p_code) # compute mf_wfvf

        for metric in m_metrics:
            metric_path = None
            if model == "dti":
                metric_path = "%s/%s_%s.nii.gz" % (model_path, p_code, metric)
            elif model in ["noddi", "diamond", "mf"]:
                metric_path = "%s/%s_%s_%s.nii.gz" % (model_path, p_code, model, metric)

            for i in range(len(densities)):
                roi_name = idx_ROIpaths[i].split("/")[-1].split("_")[2]
                addMetrics(roi_name, metric, model, metric_path, idx_ROIpaths[i])

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