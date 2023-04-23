import sys
import os
import subprocess
import json
import elikopy
import elikopy.utils

from dipy.io.streamline import load_tractogram, save_trk
from regis.core import find_transform, apply_transform
from params import get_folder, get_segmentation
from multiprocessing.pool import ThreadPool

# absolute_path = os.path.dirname(__file__) # return the abs path of the folder of this file, wherever it is

class ROI:
    def __init__(self, name, path, isCortex) -> None:
        self.name = name
        self.path = path
        self.isCortex = isCortex

    def __str__(self) -> str:
        return self.path

tracts = {
        # Non conto lo Stria-Terminalis ma devo scriverlo nella tesi che non l'ho messo.. la ragione e qualche foto
        # "stria_terminalis":
        #     {
        #         "seed_images": ["amygdala"],
        #         "include_ordered" : ["fornixST", "fornix", "BNST"],
        #         "exclude" : ["hippocampus", "Thalamus-Proper", "lateral-ventricle", "Inf-Lat-Vent", "Caudate", "Putamen", "Pallidum", "CSF", "Accumbens-area", "3rd-Ventricle"]
        #     }

        "fornix":
            {
                "seed_images": ["hippocampus"],
                "include" : ["choroid-plexus-dilated-3"],
                "include_ordered" : ["fornixST", "fornix"], 
                "exclude" : ["Thalamus-Proper", "lateral-ventricle", "Inf-Lat-Vent", "Caudate", "Putamen", "Pallidum", "CSF", "Accumbens-area", "3rd-Ventricle"],
            },

        "thalamus-AntCingCtx":
            {
                "seed_images": ["Thalamus-Proper"],
                "include_ordered" : ["aboveCC", "frontal-cingulate"],
                "exclude" : ["Lateral-Ventricle", "caudate-dilated-3", "putamen-dilated-3", "pallidum-dilated-3", "csf", "parietal-lobe", "frontal-lobe"],
                "angle" : 22
            },
        "thalamus-Insula":
            {
                "seed_images": ["Thalamus"],
                "include" : ["insula"],
                "exclude" : ["Lateral-Ventricle", "Inf-Lat-Vent", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala", "CSF", "occipital-lobe-dilated-4", "parietal-lobe-dilated-4", "frontalNoOrbito-lobe-dilated-1", "superiortemporal"],
                "angle" : 22
            },

        "sup-longi-fasci":
            { 
                "seed_images" : ["frontal-lobe"],
                "include" : ["parietal-lobe"],
                "masks" : ["cerebral-white-matter", "frontal-lobe", "parietal-lobe"],
                "angle" : 15,
                "cutoff" : 0.09
            },
        "inf-longi-fasci":
            { 
                "seed_images" : ["occipital-lobe"],
                "include" : ["temporal-lobe"],
                "masks" : ["cerebral-white-matter", "occipital-lobe", "temporal-lobe"],
                "angle" : 15,
                "cutoff" : 0.09
            },

        # Non conto l'Inferior front-occipital ma devo scriverlo nella tesi xche non l'ho messo.. la ragione e qualche foto
        # "inf-front-occipital-fasci":
        #     { 
        #         "seed_images" : ["frontal-lobe"],
        #         "include" : ["occipital-lobe"],
        #         "masks" : ["cerebral-white-matter", "frontal-lobe", "occipital-lobe"],
        #         "angle" : 15,
        #         "cutoff" : 0.09
        #     },
          }

# Freesurfer LUT: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
roi_freesurfer = {
    "hippocampus" : [17, 53],
    "amygdala" : [18, 54],
    "thalamus" : [10, 49],
    "lateral-ventricles" : [4, 43, 5, 44],
    "caudate" : [11, 50],
    "putamen" : [12, 51],
    "pallidum" : [13, 52],
    "csf" : [24],
    "accumbens" : [26, 58],
    "3rd-ventricle" : [14],
    "insula" : [1035, 2035],
    "wm" : [2, 41],
    "choroid-plexus" : [31, 63],
    "ctx-superiortemporal" : [1030, 2030],

    # Union region
    "Left-Frontal-Cingulate" : [1026, 1002],
    "Right-Frontal-Cingulate" : [2026, 2002],
    # Lobe informations taken from:
    # Freesurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation
    # Wikipedia: https://en.wikipedia.org/wiki/Association_fiber
    "Left-Frontal-Lobe" : [1028, 1027, 1003, 1018, 1019, 1020, 1012, 1014, 1024, 1017, 1032],
    "Right-Frontal-Lobe" : [2028, 2027, 2003, 2018, 2019, 2020, 2012, 2014, 2024, 2017, 2032],
    "Left-Temporal-Lobe" : [1030, 1015, 1009, 1001, 1007, 1034, 1006, 1033, 1016],
    "Right-Temporal-Lobe" : [2030, 2015, 2009, 2001, 2007, 2034, 2006, 2033, 2016],
    "Left-Parietal-Lobe" : [1008, 1029, 1031, 1022, 1025],
    "Right-Parietal-Lobe" : [2008, 2029, 2031, 2022, 2025],
    "Left-Occipital-Lobe" : [1011, 1013, 1005, 1021],
    "Right-Occipital-Lobe" : [2011, 2013, 2005, 2021],

    "Left-FrontalNoOrbito-Lobe" : [1028, 1027, 1003, 1019, 1012, 1017, 1032],
    "Right-FrontalNoOrbito-Lobe" : [2028, 2027, 2003, 2019, 2012 , 2017, 2032],
}
roi_num_name = {}
dilatations = {
    "parietal-lobe" : 4,
    "occipital-lobe" : 4,
    "frontalNoOrbito-lobe" : 1,
    "putamen" : 3,
    "caudate" : 3,
    "pallidum" : 3,
    "choroid-plexus": 3,
}

def expand_roi():
    roi_nums_tot = []
    for name, roi_numbers in roi_freesurfer.items():
        if "lobe" in name.lower() or "cingulate" in name.lower():
            continue 
        for roi_num in roi_numbers:
            roi_nums_tot.append(roi_num)
    return roi_nums_tot

def get_freesurfer_roi_names():
    colorLUT = os.getenv('FREESURFER_HOME') + "/FreeSurferColorLUT.txt"
    roi_nums_tot = expand_roi()
    roi_nums_tot.sort()
    k = 0
    
    with open(colorLUT, "r") as f:
        for line in f.readlines():
            elems = line.split()
            if len(elems) == 0:
                continue
            if elems[0] == "#" or not elems[0].isdigit():
                continue
            roi_num = int(elems[0])
            roi_name = elems[1]
            if roi_num == roi_nums_tot[k]:
                roi_num_name[roi_num] = roi_name
                k += 1
            if k == len(roi_nums_tot):
                break

def freesurfer_mask_extraction(folder_path, seg_path, subj_id):
    for num, name in roi_num_name.items():
        out_path = "%s/subjects/%s/masks/%s_%s_aparc+aseg.nii.gz" % (folder_path, subj_id, subj_id, name)
        cmd = "mri_extract_label -exit_none_found %s/%s/mri/aparc+aseg.mgz %d %s" % (seg_path, subj_id, num, out_path)
        print(cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        process.wait()
        if process.returncode != 0:
            os.remove(out_path)

    for name, roi_numbers in roi_freesurfer.items():
        if "lobe" in name.lower() or "cingulate" in name.lower():
            roi_numbers_string = " ".join(str(i) for i in roi_numbers)
            out_path = "%s/subjects/%s/masks/%s_%s_aparc+aseg.nii.gz" % (folder_path, subj_id, subj_id, name)
            cmd = "mri_extract_label -exit_none_found %s/%s/mri/aparc+aseg.mgz %s %s" % (seg_path, subj_id, roi_numbers_string, out_path)
            print(cmd)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            process.wait()
            if process.returncode != 0:
                os.remove(out_path)

    masks_path = "%s/subjects/%s/masks" % (folder_path, subj_id)
    file_extracted_names = os.listdir(masks_path)
    for file_name in file_extracted_names:
        file_path = masks_path + "/" + file_name
        ext = "."+".".join(file_name.split(".")[-2:])
        if os.path.isfile(file_path) and ext == ".nii.gz":
            roi_name = "-".join(file_name.split(subj_id + "_")[1].split("_")[0].split("-")[1:]).lower()

            for roi_dila in dilatations.keys():
                if roi_dila.lower() == roi_name.lower():
                    dilatation_cicle = dilatations[roi_dila]
                    dilated_roi_name = file_name.split(subj_id + "_")[1].split("_")[0] + "-dilated-" + str(dilatation_cicle)
                    output_path = "%s/subjects/%s/masks/%s_%s_aparc+aseg.nii.gz" % (folder_path, subj_id, subj_id, dilated_roi_name)

                    cmd = "maskfilter -force -npass %d %s dilate %s" % (dilatation_cicle, file_path, output_path)
                    print(cmd)
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                    process.wait()
                    break

def registration(folder_path, subj_id):

    moving_file_FA = folder_path + "/static_files/atlases/FSL_HCP1065_FA_1mm.nii.gz"
    moving_file_T1 = folder_path + "/static_files/atlases/MNI152_T1_05mm.nii.gz"

    static_file_FA = folder_path + "/subjects/" + subj_id + "/dMRI/microstructure/dti/" + subj_id + "_FA.nii.gz"
    static_file_T1 = folder_path + "/T1/" + subj_id + "_T1.nii.gz"

    print(moving_file_FA, moving_file_T1, static_file_FA, static_file_T1)

    if not os.path.isfile(moving_file_FA) or not os.path.isfile(static_file_FA) or not os.path.isfile(moving_file_T1) or not os.path.isfile(static_file_T1):
        print("Images for subject: " + subj_id + " weren't found")
        return 1

    mapping_FA = None
    mapping_T1 = None

    for path, dirs, files in os.walk(folder_path + "/static_files/atlases/masks"):
        for file in files: # register all the masks
            if file.endswith(".nii.gz") :

                mask_file = path + "/" + file
                moving_file = mask_file
                output_path = folder_path + "/subjects/" + subj_id + "/masks/" + subj_id + "_" + file

                print("Applying transformation from " + file.split(".")[0])
                if "_FA_" in file:
                    
                    if mapping_FA is None:
                        print("Finding transformation from atlas FA ")
                        mapping_FA = find_transform(moving_file_FA, static_file_FA)
                        print("Transformation found")

                    print("FA:",end=" ")

                    apply_transform(moving_file, mapping_FA, static_file_FA, output_path=output_path, binary=True, binary_thresh=0)
                elif "_T1_" in file:
                    
                    if mapping_T1 is None:
                        print("Finding transformation from atlas T1")
                        mapping_T1 = find_transform(moving_file_T1, static_file_T1)
                        print("Transformation found")
                    
                    print("T1:",end=" ")

                    apply_transform(moving_file, mapping_T1, static_file_T1, output_path=output_path, binary=True, binary_thresh=0)
                print("Transformed")

def get_mask(mask_path, subj_id):
    roi_names = {}
    roi_names["left"] = {}
    roi_names["right"]= {}

    for path, _, files in os.walk(mask_path):
        for file in files:
            fileName_ext = file.split(".")
            fileName = fileName_ext[0]
            ext = "."+".".join(fileName_ext[1:])
            if ext != ".nii.gz" :
                continue
    
            roiName = fileName.split(subj_id+"_")[1].split("_")[0].lower().split("-")
            if "left" not in roiName and "right" not in roiName and "lh" not in roiName and "rh" not in roiName and "both" not in roiName:
                if roiName[0] == "csf" or len(roiName) >= 2 :
                    roiName.insert(0, "both")
                else :
                    continue
            if len(roiName) < 2:
                    continue
            name = None; side = None; isCortex = None

            if "ctx" != roiName[0]:
                side, name = roiName[0], "-".join(roiName[1:])
                isCortex = False
            else :
                side, name = roiName[1], "-".join(roiName[2:])
                side = "left" if side == "lh" else "right"
                isCortex = True
            
            if side == "both":
                roi_names["left"][name] = ROI(name, path+"/"+file, isCortex)
                roi_names["right"][name] = ROI(name, path+"/"+file, isCortex)
                continue

            roi_names[side][name] = ROI(name, path+"/"+file, isCortex)

    return roi_names

def find_tract(subj_folder_path, subj_id, seed_images, inclusions, inclusions_ordered, exclusions, masks, angle, cutoff, stop, act, output_name):
    """
    It's a function that build the bashCommands for the tckgen of mrtrix3 and generate the tracts
    """
    tck_path = subj_folder_path+"/dMRI/tractography/"+output_name+".tck"
    process = None

    bashCommand = "tckgen -nthreads 4 -algorithm iFOD2 -seeds 1M -max_attempts_per_seed 1000 -seed_unidirectional -force"

    if stop:
        bashCommand += " -stop"
    if act:
        bashCommand += " -act " + subj_folder_path + "/dMRI/5tt/subj00_5tt.nii.gz"

    bashCommand += " -angle " + str(angle)
    bashCommand += " -cutoff " + str(cutoff)
    bashCommand += (" -fslgrad " + subj_folder_path + "/dMRI/preproc/"+subj_id+"_dmri_preproc.bvec " +
                                   subj_folder_path + "/dMRI/preproc/"+subj_id+"_dmri_preproc.bval")
    
    for region in seed_images:
        bashCommand += " -seed_image " + region
    for region in inclusions:
        bashCommand += " -include " + region
    for region in inclusions_ordered:
        bashCommand += " -include_ordered " + region
    for region in exclusions:
        bashCommand += " -exclude " + region
    for region in masks:
        bashCommand += " -mask " + region

    bashCommand += " " + subj_folder_path + "/dMRI/ODF/MSMT-CSD/"+subj_id+"_MSMT-CSD_WM_ODF.nii.gz " + tck_path
    
    print(bashCommand)
    print(" ")

    process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
    process.wait()

    return tck_path

def convertTck2Trk(subj_folder_path, subj_id, tck_path):
    if not os.path.isfile(tck_path): # only if there was an error during the tractography, to not block everything
        return
    print("Converting %s" % tck_path)
    tract = load_tractogram(tck_path, subj_folder_path+"/dMRI/preproc/"+subj_id+"_dmri_preproc.nii.gz")
    save_trk(tract, tck_path[:-3]+'trk')

def compute_tracts(p_code, folder_path, extract_roi, seg_path):
    print("Working on %s" % p_code)

    subj_folder_path = folder_path + '/subjects/' + p_code
        
    # check if the ODF exist for the subject, otherwise skip subject
    if not os.path.isdir(subj_folder_path + "/dMRI/ODF/MSMT-CSD/") :
        print("multi-tissue orientation distribution function is not found for patient: %s" % (p_code))
        return 1

    if not os.path.isdir(subj_folder_path + "/dMRI/tractography/"):
        os.mkdir(subj_folder_path + "/dMRI/tractography/")
    if not os.path.isdir(subj_folder_path + "/masks/"):
        os.mkdir(subj_folder_path + "/masks/")

    ############# ROI EXTRACTION ############

    if extract_roi:

        # Do the registration to extract ROI from atlases
        print("Registration on %s" % p_code)
        if registration(folder_path, p_code) is not None:
            return 1

        # Extract ROI from freesurfer segmentation
        # check if the freesurfer segmentation exist, otherwise skip subject
        # Here we are assuming that the segmentation is already done
        if not os.path.isdir(seg_path + "/" + p_code + "/mri"):
            print("freesurfer segmentation isn't found for paritent: %s" % (p_code))
            return 1

        print("Freesurfer roi extraction on %s" % p_code)
        freesurfer_mask_extraction(folder_path, seg_path, p_code)

    roi_names = get_mask(subj_folder_path+"/masks", p_code)

    ########### TRACTOGRAPHY ##########
    for zone in tracts.keys():
        for side in ["left", "right"]:
            opts = {}

            opts["seed_images"] = []
            opts["include"] = []
            opts["include_ordered"] = []
            opts["exclude"] = []
            opts["masks"] = []
            opts["angle"] = 45
            opts["cutoff"] = 0.1
            opts["stop"] = True
            opts["act"] = False

            areAllROIs = True

            # convert the option in path of the associated file
            for opt, rois in tracts[zone].items():
                if type(rois) is list:
                    for roi in rois:
                        # find the file name inside the roi_names
                        if roi.lower() not in roi_names[side]:
                            print("Mask of roi %s isn't found: skipping %s" % (roi.lower(), zone))
                            areAllROIs = False
                            continue
                        opts[opt].append(roi_names[side][roi.lower()].path)
                elif type(rois) is int or type(rois) is float or type(rois) is bool:
                    opts[opt] = rois
            
            if not areAllROIs: # All the mask must be present
                continue

            if len(opts["include_ordered"]) > 0 and len(opts["seed_images"]) > 1:
                print("The case of more seed_regions and an ordered list of regions is not handled by this program")
                continue

            output_name = side+"-"+zone
            output_path = subj_folder_path+"/dMRI/tractography/"+output_name+".tck"

            # TODO Check for act files (5TT) !!!!!!! DOESN'T WORK !!!!!!!!!
            path_5tt = subj_folder_path + "/dMRI/5tt/subj00_5tt.nii.gz"
            if opts["act"] == True and not os.path.isfile(path_5tt):
                seg_path = get_segmentation(sys.argv)
                colorLUT = os.getenv('FREESURFER_HOME') + "/FreeSurferColorLUT.txt"
                bashCommand = "5ttgen freesurfer %s %s" % (seg_path + "/" + p_code + "/mri/aparc+aseg.mgz", path_5tt, colorLUT)
                print(bashCommand)
                process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
                process.wait()

            # forward
            output_path_forward = find_tract(subj_folder_path, p_code, opts["seed_images"], opts["include"], opts["include_ordered"], opts["exclude"], opts["masks"], opts["angle"], opts["cutoff"], opts["stop"], opts["act"], output_name+"_to.tmp")

            optsReverse = {}
            if len(opts["include_ordered"]) == 0: 
                optsReverse["seed_images"] = opts["include"]
                optsReverse["include"] = opts["seed_images"]
                optsReverse["include_ordered"] = []
            elif len(opts["seed_images"]) == 1:
                optsReverse["seed_images"] = []
                optsReverse["seed_images"].append(opts["include_ordered"][-1])
                optsReverse["include_ordered"] = opts["include_ordered"][::-1][1:]
                optsReverse["include_ordered"].extend(opts["seed_images"])
                optsReverse["include"] = opts["include"]

            # backward
            output_path_backward = find_tract(subj_folder_path, p_code, optsReverse["seed_images"], optsReverse["include"], optsReverse["include_ordered"], opts["exclude"], opts["masks"], opts["angle"], opts["cutoff"], opts["stop"], opts["act"], output_name+"_from.tmp")

            # select both tracks 
            if os.path.isfile(output_path_forward) and os.path.isfile(output_path_backward):
                cmd = "tckedit -force %s %s %s" % (output_path_forward, output_path_backward, output_path)
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                process.wait()

                os.remove(output_path_forward); os.remove(output_path_backward)

                convertTck2Trk(subj_folder_path, p_code, output_path)

NTHREADS = 5

def main():

    ## Getting folder
    folder_path = get_folder(sys.argv)
    
    # check if the user wants to compute the ODF and compute it
    if "-odf" in sys.argv[1:]:
        study = elikopy.core.Elikopy(folder_path, cuda=True, slurm=True, slurm_email="michele.cerra@student.uclouvain.be")

        study.odf_msmtcsd()

    extract_roi = False
    seg_path = ""
    if "-roi" in sys.argv[1:]:
        extract_roi = True  
        seg_path = get_segmentation(sys.argv)
        get_freesurfer_roi_names()

    ## Read the list of subjects and for each subject do the tractography
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as file:
        patient_list = json.load(file)

    with ThreadPool(NTHREADS) as pool:
        args = [(p, folder_path, extract_roi, seg_path) for p in patient_list]
        pool.starmap(compute_tracts, args)

if __name__ == "__main__":
    exit(main())
