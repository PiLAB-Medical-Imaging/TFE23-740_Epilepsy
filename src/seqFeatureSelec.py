import sys
import pandas as pd
import matplotlib.pyplot as plt

from params import *
from elikopy.utils import submit_job
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.pipeline import make_pipeline

def readingDF(stats_path):
    # Reading the whole dataset
    df = pd.read_csv("%s/dataset.csv" % stats_path, index_col="ID")
    print(df.shape)
    # # Removing the healty subjects
    # df.dropna(axis=0, how="any", inplace=True) 
    # print(df.shape)

    info = df.columns[:10]
    df[info] = df[info].astype(int)

    df = df.drop("VNSLC_16") # remove it because doesn't have the tracts from freesurfer
    df = df.dropna(axis=1) # remove NaN features
    df = df.drop(df.filter(regex=r'(min|max)'), axis=1) # remove max and min features because are full of imperfections
    df = df.drop(df.filter(regex=r'(kurt|nTracts)'), axis=1) # remove them beacuse they have an high variability and difficult to interpretare (only kurt)
    df = df.loc[:, (abs(df - df.iloc[0]) > 1e-12).any()] # Removing almost constant features
    print(df.shape)

    return df

def seqFeatureSelecFunc(stats_path):
    df = readingDF(stats_path)

    X = df.drop(["resp", "respPart"], axis=1)
    y = df["resp"]

    pipe = make_pipeline(
        StandardScaler(),
        SGDClassifier(loss = 'log_loss',
                             n_jobs = -1, 
                             penalty = 'l2', 
                             alpha=50,
                             max_iter=10000
        )
    )

    selector = SFS(
        pipe,
        k_features=(2, 10),
        forward=True,
        floating=True,
        scoring="f1",
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
    plt.savefig(stats_path+"/SFS_forward3.png")

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
        "time" : "08:00:00",
        "mail_user" : "michele.cerra@student.uclouvain.be",
        "mail_type" : "FAIL",
        "output" : stats_path + f"/slurm-%j.out",
        "error" : stats_path + f"/slurm-%j.err",
    }
    submit_job(p_job)


if __name__=="__main__":
    exit(main())
