import json
import os
import pandas as pd

def concat():
    folder_path = "../study"
    stats_path = folder_path + "/stats"

    ## Read the list of subjects and for each subject do the tractography
    dest_success = folder_path + "/subjects/subj_list.json"
    with open(dest_success, 'r') as file:
        patient_list = json.load(file)
    del file

    dfs = []
    col_names = None
    for p_code in patient_list:
        metric_folder = "%s/subjects/%s/dMRI/microstructure/%s_metrics.csv" % (folder_path, p_code, p_code)
        if not os.path.exists(metric_folder):
            print(metric_folder, "doesn't exists")
            continue
        dfi = pd.read_csv(metric_folder)
        dfi = dfi.reindex(sorted(dfi.columns), axis=1)
        # Check that the name of the columns are correct
        # if dfi.columns.size != 7518 + 1:
        print(p_code, dfi.columns.size)

        dfs.append(dfi)
        del metric_folder
    del p_code

    df = pd.concat(dfs, ignore_index=True, axis=0)
    del dfs

    # assert df.columns.size == 7518 + 1

    info_df = pd.read_csv(stats_path + "/info.csv")

    df = pd.merge(info_df, df, on="ID")
    df = df.set_index("ID")
    del info_df

    if not os.path.isdir(stats_path):
        os.mkdir(stats_path)

    df.to_csv("%s/datasetRadiomics.csv" % stats_path)

if __name__=="__main__":
    exit(concat())