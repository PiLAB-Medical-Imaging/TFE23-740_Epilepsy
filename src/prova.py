import os

mask_path = "../study/subjects/subj00/masks"

tracts_roi = {
    "fornix" : "17 53",
    "stria_terminalis" : "18 54",
    "cortex" : "10 49",
    "cortex_interval1" : "1001 1035",
    "cortex_interval2" : "2001 2035"
}

tracts_roi.values()

roi_names = []
ctx_names = []

roi_names.pop

for path, dirs, files in os.walk(mask_path):
    dir_name = path.split("/")[-1]
    if dir_name in tracts_roi:
        print(path, dirs, files)
        for file in files:
            no_ext = file.split(".")[0]
            no_subj = "_".join(no_ext.split("_")[1:])
            roi_names.append(no_subj)
            if "ctx" in no_subj:
                ctx_names.append(no_subj)

for roi in ctx_names:
    print(roi)
