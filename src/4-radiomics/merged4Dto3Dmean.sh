## Don't used ini the thesis, but can be useful

for i in $(ls -la ../study/freesurfer/VNSLC_*/dpath/merged_avg16_syn_bbr.mgz | cut -d " " -f 10); do
    echo $i
    new_name="$(dirname $i)/mergedX2_3D_avg16_syn_bbr.nii.gz"
    mrmath $i mean $new_name -axis 3
done