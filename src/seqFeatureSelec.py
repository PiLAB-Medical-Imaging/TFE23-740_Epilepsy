import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from params import *
from elikopy.utils import submit_job
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.pipeline import make_pipeline

from tools_analysis import auc_and_f1

def readingDF(stats_path):
    # Reading the whole dataset
    df = pd.read_csv("%s/dataset_thres1.csv" % stats_path, index_col="ID")
    print(df.shape)
    # # Removing the healty subjects
    # df.dropna(axis=0, how="any", inplace=True) 
    # print(df.shape)

    info = df.columns[:10]
    df[info] = df[info].astype(int)

    df = df.drop("VNSLC_16") # remove it because doesn't have the tracts from freesurfer
    df = df.dropna(axis=1) # remove NaN features
    df = df.drop(df.filter(regex=r'(min|max)'), axis=1) # remove max and min features because are full of imperfections
    df = df.drop(df.filter(regex=r'(_c0_|_c1_|_f0_|_f1_|std|skew|kurt|nTracts)'), axis=1) # remove them beacuse they have an high variability and difficult to interpretare (only kurt)
    df = df.loc[:, (abs(df - df.iloc[0]) > 1e-12).any()] # Removing almost constant features
    print(df.shape)

    return df

from sklearn.metrics import make_scorer
def runSelection(X, y, stats_path, pipe, nFeatures, scoreFunc, model_name):
    selector = SFS(
        pipe,
        k_features=nFeatures,
        forward=True,
        floating=False,
        scoring=make_scorer(auc_and_f1, needs_proba=True, needs_threshold=True),
        cv=StratifiedShuffleSplit(n_splits=1000, test_size=1/3, random_state=7),
        n_jobs=-1,
        verbose=2,   
    )

    selector = selector.fit(X, y)

    print('Best accuracy score: %.2f' % selector.k_score_)
    print('Best subset (indices):', selector.k_feature_idx_)
    print('Best subset (corresponding names):', selector.k_feature_names_)

    fig1 = plot_sfs(selector.get_metric_dict(), kind='std_dev')

    plt.ylim([0.8, 1])
    plt.title('Sequential Forward Selection (w. StdDev)')
    plt.grid()
    plt.savefig(stats_path+f"/SFS_forward_{model_name}_{scoreFunc}_{str(nFeatures)}.png")

    print()

def seqFeatureSelecFunc(stats_path):
    df = readingDF(stats_path)

    X = df.drop(["resp", "respPart"], axis=1)
    y = df["resp"]

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            dual=True,
            class_weight="balanced",
            C=,
            random_state=7,
            solver="liblinear",
            max_iter=100000,
        )
    )
    model_name = "log_reg"
    runSelection(X, y, stats_path, pipe, 2, "roc_auc", model_name)
    runSelection(X, y, stats_path, pipe, 3, "roc_auc", model_name)

    pipe = make_pipeline(
        StandardScaler(),
        LinearSVC(
            penalty="l2",
            loss="hinge",
            dual=True,
            C=5e-3,
            class_weight="balanced",
            random_state=7,
            max_iter=100000,
        )
    )
    model_name = "svm"
    runSelection(X, y, stats_path, pipe, 2, "roc_auc", model_name)
    runSelection(X, y, stats_path, pipe, 3, "roc_auc", model_name)

def main():
    ## Getting folder
    folder_path = get_folder(sys.argv)
    stats_path = folder_path + "/stats"

    # seqFeatureSelec(stats_path)

    p_job = {
        "wrap" : f"export MKL_NUM_THREADS=24 ; export OMP_NUM_THREADS=24 ; python -c 'from seqFeatureSelec import seqFeatureSelecFunc; seqFeatureSelecFunc(\"{stats_path}\")'",
        "job_name" : "SFS",
        "ntasks" : 1,
        "cpus_per_task" : 24,
        "mem_per_cpu" : 1024,
        "time" : "5:00:00",
        "mail_user" : "michele.cerra@student.uclouvain.be",
        "mail_type" : "FAIL",
        "output" : stats_path + f"/slurm-%j.out",
        "error" : stats_path + f"/slurm-%j.err",
    }
    submit_job(p_job)


if __name__=="__main__":
    exit(main())
