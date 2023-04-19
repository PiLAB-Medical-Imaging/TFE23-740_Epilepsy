from unravel.utils import tract_to_ROI, load_tractogram, get_streamline_density
import nibabel as nib

fa_map = "/home/michele/Dropbox (Politecnico Di Torino Studenti)/thesis/code/Epilepsy-dMRI-VNS/study/subjects/subj00/dMRI/microstructure/dti/subj00_FA.nii.gz"
img  = nib.load(fa_map)

data = img.get_fdata()

mask = tract_to_ROI("/home/michele/Dropbox (Politecnico Di Torino Studenti)/thesis/code/Epilepsy-dMRI-VNS/study/subjects/subj00/dMRI/tractography/right-fornix.trk")


out = nib.Nifti1Image(mask, img.affine)

out.to_filename("./trkMask.nii.gz")


trk = load_tractogram("/home/michele/Dropbox (Politecnico Di Torino Studenti)/thesis/code/Epilepsy-dMRI-VNS/study/subjects/subj00/dMRI/tractography/right-fornix.trk", 'same')

trk.to_vox()
trk.to_corner()

density_map = get_streamline_density(trk)

out = nib.Nifti1Image(density_map, img.affine)
out.to_filename("./trkMaskDensity.nii.gz")
