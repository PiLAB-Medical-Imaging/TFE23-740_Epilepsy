import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from params import *
from elikopy.utils import submit_job
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.pipeline import make_pipeline

from tools_analysis import auc_and_f1

def readingDF(stats_path):
    # Reading the whole dataset
    df = pd.read_csv("%s/dataset_thres1_1corr.csv" % stats_path, index_col="ID")
    print(df.shape, flush=True)
    # # Removing the healty subjects
    # df.dropna(axis=0, how="any", inplace=True) 
    # print(df.shape)

    info = df.columns[:10]
    df[info] = df[info].astype(int)

    df = df.drop("VNSLC_16") # remove it because doesn't have the tracts from freesurfer
    df = df.dropna(axis=1) # remove NaN features
    df = df.drop(df.filter(regex=r'(min|max)'), axis=1) # remove max and min features because are full of imperfections
    df = df.drop(df.filter(regex=r'(_c0_|_c1_|_f0_|_f1_|_csf_mf_|_csf_d_|_fiso_|nTracts|voxVol)'), axis=1)
    df = df.loc[:, (abs(df - df.iloc[0]) > 1e-12).any()] # Removing almost constant features
    print(df.shape, flush=True)

    X = df.drop(["resp", "respPart"], axis=1)
    # X = X.filter(regex=r'(mean|age|duration|sex)')
    X = X.filter(regex=r'(mean|std|skew|kurt|age|duration|sex)')
    y = df["respPart"]

    return X, y

from sklearn.metrics import make_scorer
def runSelection(X, y, stats_path, pipe, nFeatures, model_name):
    selector = SFS(
        pipe,
        k_features=nFeatures,
        forward=True,
        floating=True,
        scoring=make_scorer(auc_and_f1, needs_proba=True),
        cv=StratifiedShuffleSplit(n_splits=1000, test_size=1/3, random_state=7),
        n_jobs=-1,
        verbose=2,   
    )

    selector = selector.fit(X, y)

    print('Best accuracy score: %.2f' % selector.k_score_, flush=True)
    print('Best subset (indices):', selector.k_feature_idx_, flush=True)
    print('Best subset (corresponding names):', selector.k_feature_names_, flush=True)

    fig1 = plot_sfs(selector.get_metric_dict(), kind='std_dev')

    plt.ylim([0, 2])
    plt.title('Sequential Forward Selection (w. StdDev)')
    plt.grid()
    plt.savefig(stats_path+f"/SFS_forward_3_{model_name}_{str(nFeatures)}.png")

    print()

def SFSlogreg(stats_path):
    X, y = readingDF(stats_path)

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=5e-6,
            class_weight="balanced",
            random_state=7,
            solver="lbfgs",
            multi_class="multinomial",
            max_iter=100000,
        )
    )
    model_name = "log_reg"
    runSelection(X, y, stats_path, pipe, 2, model_name)
    runSelection(X, y, stats_path, pipe, 3, model_name)
    runSelection(X, y, stats_path, pipe, 10, model_name)

def SFSSVM(stats_path, kernel, degree : int, C, gamma):
    X, y = readingDF(stats_path)

    pipe = SVC(
        probability=True,
        C=C,
        gamma=gamma,
        kernel=kernel,
        degree=degree,
        class_weight="balanced",
        decision_function_shape="ovo",
        max_iter=-1,
        random_state=7,
    )
    model_name = f"svm_{kernel}_{degree}"
    runSelection(X, y, stats_path, pipe, 2, model_name)
    runSelection(X, y, stats_path, pipe, 3, model_name)
    runSelection(X, y, stats_path, pipe, 10, model_name)

def SFSgaussian(stats_path):
    X, y = readingDF(stats_path)

    pipe = make_pipeline(
        StandardScaler(),
        GaussianNB()
    )
    model_name = "gaussian"
    runSelection(X, y, stats_path, pipe, 2, model_name)
    runSelection(X, y, stats_path, pipe, 3, model_name)
    runSelection(X, y, stats_path, pipe, 10, model_name)

def SFSKNN(stats_path):
    X, y = readingDF(stats_path)

    pipe = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=3,
            weights="distance",
            n_jobs=-1,
        )
    )
    model_name = "knn"
    runSelection(X, y, stats_path, pipe, 2, model_name)
    runSelection(X, y, stats_path, pipe, 3, model_name)
    runSelection(X, y, stats_path, pipe, 10, model_name)

def SFStree(stats_path):
    X, y = readingDF(stats_path)

    pipe = make_pipeline(
        StandardScaler(),
        DecisionTreeClassifier(
            criterion="gini",
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=2,
            min_weight_fraction_leaf=0.25,
            random_state=7,
            class_weight="balanced",
        )
    )
    model_name = "decisionTree"
    runSelection(X, y, stats_path, pipe, 2, model_name)
    runSelection(X, y, stats_path, pipe, 3, model_name)
    runSelection(X, y, stats_path, pipe, 10, model_name)

def main():
    ## Getting folder
    folder_path = get_folder(sys.argv)
    stats_path = folder_path + "/stats"

    p_job = {
        "wrap" : f"export MKL_NUM_THREADS=12 ; export OMP_NUM_THREADS=12 ; python -c 'from seqFeatureSelec3 import SFSlogreg; SFSlogreg(\"{stats_path}\")'",
        "job_name" : "3logreg",
        "ntasks" : 1,
        "cpus_per_task" : 12,
        "mem_per_cpu" : 512,
        "time" : "8:00:00",
        "mail_user" : "michele.cerra@student.uclouvain.be",
        "mail_type" : "FAIL",
        "output" : stats_path + f"/logreg_3_all.out",
        "error" : stats_path + f"/logreg_3.err",
    }
    submit_job(p_job)

    p_job = {
        "wrap" : f"export MKL_NUM_THREADS=12 ; export OMP_NUM_THREADS=12 ; python -c 'from seqFeatureSelec3 import SFSSVM; SFSSVM(\"{stats_path}\", \"linear\", 1, 5e-3, \"scale\")'",
        "job_name" : "3svmlin",
        "ntasks" : 1,
        "cpus_per_task" : 12,
        "mem_per_cpu" : 512,
        "time" : "8:00:00",
        "mail_user" : "michele.cerra@student.uclouvain.be",
        "mail_type" : "FAIL",
        "output" : stats_path + f"/SVMlin_3_all.out", 
        "error" : stats_path + f"/SVMlin_3.err",
    }
    submit_job(p_job)

    p_job = {
        "wrap" : f"export MKL_NUM_THREADS=12 ; export OMP_NUM_THREADS=12 ; python -c 'from seqFeatureSelec3 import SFSSVM; SFSSVM(\"{stats_path}\", \"poly\", 3, 1, \"scale\")'",
        "job_name" : "3svmpoly3",
        "ntasks" : 1,
        "cpus_per_task" : 12,
        "mem_per_cpu" : 512,
        "time" : "8:00:00",
        "mail_user" : "michele.cerra@student.uclouvain.be",
        "mail_type" : "FAIL",
        "output" : stats_path + f"/SVMpoly3_3_all.out", 
        "error" : stats_path + f"/SVMpoly3_3.err",
    }
    submit_job(p_job)
    
    p_job = {
        "wrap" : f"export MKL_NUM_THREADS=12 ; export OMP_NUM_THREADS=12 ; python -c 'from seqFeatureSelec3 import SFSSVM; SFSSVM(\"{stats_path}\", \"rbf\", 0, 0.001, 0.01)'",
        "job_name" : "3svmrbf",
        "ntasks" : 1,
        "cpus_per_task" : 12,
        "mem_per_cpu" : 512,
        "time" : "8:00:00",
        "mail_user" : "michele.cerra@student.uclouvain.be",
        "mail_type" : "FAIL",
        "output" : stats_path + f"/SVMrbf_3_all.out",
        "error" : stats_path + f"/SVMrbf_3.err",
    }
    submit_job(p_job)

    p_job = {
        "wrap" : f"export MKL_NUM_THREADS=12 ; export OMP_NUM_THREADS=12 ; python -c 'from seqFeatureSelec3 import SFSgaussian; SFSgaussian(\"{stats_path}\")'",
        "job_name" : "3naive",
        "ntasks" : 1,
        "cpus_per_task" : 12,
        "mem_per_cpu" : 512,
        "time" : "8:00:00",
        "mail_user" : "michele.cerra@student.uclouvain.be",
        "mail_type" : "FAIL",
        "output" : stats_path + f"/gaussian_3_all.out",
        "error" : stats_path + f"/gaussian_3.err",
    }
    submit_job(p_job)

    p_job = {
        "wrap" : f"export MKL_NUM_THREADS=12 ; export OMP_NUM_THREADS=12 ; python -c 'from seqFeatureSelec3 import SFSKNN; SFSKNN(\"{stats_path}\")'",
        "job_name" : "3knn",
        "ntasks" : 1,
        "cpus_per_task" : 12,
        "mem_per_cpu" : 512,
        "time" : "8:00:00",
        "mail_user" : "michele.cerra@student.uclouvain.be",
        "mail_type" : "FAIL",
        "output" : stats_path + f"/KNN_3_all.out",
        "error" : stats_path + f"/KNN_3.err",
    }
    submit_job(p_job)
     
    p_job = {
        "wrap" : f"export MKL_NUM_THREADS=12 ; export OMP_NUM_THREADS=12 ; python -c 'from seqFeatureSelec3 import SFStree; SFStree(\"{stats_path}\")'",
        "job_name" : "3tree",
        "ntasks" : 1,
        "cpus_per_task" : 12,
        "mem_per_cpu" : 512,
        "time" : "8:00:00",
        "mail_user" : "michele.cerra@student.uclouvain.be",
        "mail_type" : "FAIL",
        "output" : stats_path + f"/decisionTree_3_all.out",
        "error" : stats_path + f"/decisionTree_3.err",
    }
    submit_job(p_job)


if __name__=="__main__":
    exit(main())
