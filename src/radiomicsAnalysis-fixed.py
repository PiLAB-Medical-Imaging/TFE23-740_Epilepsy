#####
# Classification of Responders and Non-responders with 
#####

#%% Import
# Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import fastcluster
import json
import utils

from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, LeaveOneOut, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.decomposition import PCA
from statsmodels import robust
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from tqdm import tqdm

#%% Init
df = utils.getReducedDS()
X, y, y3 = utils.splitFeatureLabels(df)
#%% Split and remove outliers
X_train, X_test, y_train, y_test, y3_train, y3_test = utils.splitTrainTestDF(X, y, y3, 0.4)
y3_train = df.iloc[:, 1][y_train.index]
y3_test = df.iloc[:, 1][y_test.index]

mad_outliers = utils.MADOutlierRemotion(3)
mad_outliers.fit(X_train)

X_train = mad_outliers.transform(X_train)
X_test = mad_outliers.transform(X_test)

#%% selection

def getGoodFeatures(path):

    with open(path) as infile:
        res = json.load(infile)
    res_df = pd.DataFrame(res).T
    res_df = res_df.dropna()
    res_df =  res_df[(res_df["acc"]>0.6)]

    return res_df

#%%

getGoodFeatures("../study/stats/results-gauss-fix40-3cv.json").sort_values(by=[ "log", "auc", "acc"], ascending=[False, False, False])

#%% Combinatorial selection
# Combinatorial selection

def run_cv(X, y, selected_groups):
    print(X.shape)

    print(selected_groups)
    cv = cross_validate(
        pipe,
        X, y,
        scoring=make_scorer(retProbs, needs_proba=True),
        cv=LeaveOneOut(),
        n_jobs=-1,
        verbose=5,
        return_estimator=True,
        error_score=0.5
    )

    res[selected_groups] = printScores(y, cv["test_score"])
    print()

def extract_df(X, groups):
    X_extracted = None
    for group in groups:
        X_curr = X.filter(regex=group)
        if X_extracted is None:
            X_extracted = X_curr
        else:
            X_extracted = pd.concat([X_extracted, X_curr], axis=1)
    
    return X_extracted

def f(v, i):
    if i == v.size:
        print(v)
        if v.sum() < 1: 
            return

        selected_groups_join = ";".join(selected_groups[mask])
        res[selected_groups_join] = {}
        run_cv(extract_df(X_train, selected_groups[mask]), y_train, selected_groups_join)
        return

    v[i] = True
    f(v, i+1)
    v[i] = False
    f(v, i+1)
    return

for name, algorithm, selected_groups in [
    (
        "logreg",
        LogisticRegression(random_state=7, max_iter=500),
        ["_exponential_.*Median$", "_exponential_.*Mean$", "_wavelet-HLH_.*10Percentile$", "_wavelet-LLL_.*Minimum$"]
    ),
    (
        "svm",
        CalibratedClassifierCV(SVC(random_state=7), cv=3),
        ["_gradient_.*Kurtosis$", "_wavelet-HHL_.*DifferenceEntropy$", "_wavelet-HHL_.*DifferenceVariance$"]
    ),
    # ("knn", CalibratedClassifierCV(KNeighborsClassifier(), cv=3)),
    (
        "mlp", 
        MLPClassifier(random_state=7, alpha=1, learning_rate="adaptive", max_iter=1000,),
        ["_wavelet-LLL_.*RootMeanSquared$", "_exponential_.*Median$", "_logarithm_.*Mean$", "_wavelet-HLH_.*10Percentile$"]
    ),
    # ("gauss", GaussianNB())
]:
    selected_groups = np.array(selected_groups)
    mask = np.zeros(selected_groups.shape).astype(bool)

    pipe = Pipeline([
        ("scaler", MyScaler()),
        ("clf", algorithm)
    ])

    res = {}
    
    f(mask, 0)

    with open(f"../study/stats/results-{name}-comb-fix40.json", "w") as outfile:
        json.dump(res, outfile, indent=2, sort_keys=True)

#%% SFS 

selected_features = {}

for name, algorithm in [
    ("logreg", LogisticRegression(C=1e-6, random_state=7, max_iter=10000, class_weight="balanced")),
    ("svm", SVC(C=1e-6, random_state=7, class_weight="balanced")),
    ("knn", CalibratedClassifierCV(KNeighborsClassifier(), cv=3)),
    # ("mlp", MLPClassifier(random_state=7, alpha=2, learning_rate="adaptive", max_iter=1000)),
    # ("gauss", GaussianNB())
]:
    
    decision = True if name=="svm" else False

    selected_features[name] =  []
    for group in tqdm(getGoodFeatures(f"../study/stats/results-{name}-fix20-3cv.json").sort_values(by="acc").index):

        if len(selected_features[name]) == 100:
            break

        regex = group.replace("_", "_.*_")

        X_filtered = X_train.filter(regex=regex)

        sfs = SFS(
            Pipeline([
                ("scaler", RobustScaler()),
                ("clf", algorithm)
            ]),
            k_features=(1,X_filtered.shape[1]),
            floating=True,
            scoring="balanced_accuracy",
            cv=3,
            n_jobs=-1
        )

        sfs.fit(X_filtered, y_train)

        selected_features[name] = [*selected_features[name], *sfs.k_feature_names_]

    with open(f"../study/stats/results-{name}-fix20-3cv.selected.json", "w") as outfile:
        json.dump(selected_features[name], outfile, indent=2, sort_keys=True)


#%%
split = 40
sfss = {}

for name, algorithm in [
    ("logreg", LogisticRegression(C=1e-6, random_state=7, max_iter=10000, class_weight="balanced")),
    ("svm", SVC(C=1e-6, random_state=7, class_weight="balanced")),
    ("knn", CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=3), cv=2)),
    ("mlp", MLPClassifier(random_state=7, alpha=2, learning_rate="adaptive", max_iter=1000)),
    ("gauss", GaussianNB())
]:
    decision = True if name=="svm" else False

    s = getGoodFeatures(f"../study/stats/results-{name}-fix{split}-3cv.json").sort_values(by=["log", "brier", "auc", "acc"] if not decision else ["auc","acc"], ascending=[False, False, False, False] if not decision else [False, False])[:5].index
    regex = "|".join(s).replace("_", "_.*_")
    
    X_selected = X_train.filter(regex=regex)

    sfs = SFS(
        Pipeline([
            ("scaler", RobustScaler()),
            ("clf", algorithm)
        ]),
        k_features=2,
        floating=True,
        scoring="neg_log_loss" if not decision else "roc_auc",
        cv=3,
        verbose=1,
        n_jobs=4
    )

    sfs.fit(X_selected, y_train)

    sfss[name] = sfs.k_feature_names_

    with open(f"../study/stats/results-{name}-fix{split}-3cv.selected.final.json", "w") as outfile:
        json.dump(sfss[name], outfile, indent=2, sort_keys=True)



#%% Best Log Reg

utils.fitTrain_scoreTest(
    LogisticRegression(C=1e-6, random_state=7, max_iter=1000, class_weight="balanced"),
    X_train, X_test, y_train, y_test,
    # "Right-Accumbens-area_wavelet-LLL_Coarseness".replace("_", "_.*_"),
    "|".join(sfss["logreg"])
)

#%% Best SVM

utils.fitTrain_scoreTest(
    SVC(C=1, random_state=7, class_weight="balanced"),
    X_train, X_test, y_train, y_test,
    # "Right-Pallidum_wavelet-HLH_SmallAreaHighGrayLevelEmphasis".replace("_", "_.*_")
    "|".join(sfss["svm"])
)

#%% Best KNN

utils.fitTrain_scoreTest(
    CalibratedClassifierCV(KNeighborsClassifier(), cv=3),
    X_train, X_test, y_train, y_test,
    # "rh.or_wavelet-LHL_ClusterProminence".replace("_", "_.*_"),
    "|".join(sfss["knn"])
)

#%% Best MLP 

utils.fitTrain_scoreTest(
    MLPClassifier(random_state=7, alpha=2, learning_rate="adaptive", max_iter=1000),
    X_train, X_test, y_train, y_test,
    # "cc.rostrum_wavelet-HHL_LongRunLowGrayLevelEmphasis".replace("_", "_.*_"),
    "|".join(sfss["mlp"])
)

#%% Best Gauss

utils.fitTrain_scoreTest(
    GaussianNB(),
    X_train, X_test, y_train, y_test,
    # "Left-Hippocampus_wavelet-HLH_RobustMeanAbsoluteDeviation".replace("_", "_.*_"),
    "|".join(sfss["gauss"])
)

#%% PCA

def printPCA_train_test(DTR, DTE, LTR, LTE, regex=".*", idx=None):

    DTR_filtered = DTR.filter(regex=regex)
    DTE_filtered = DTE.filter(regex=regex)

    if DTR_filtered.shape[1] == 0:
        return None
    
    if idx is not None:
        DTR_filtered = DTR_filtered.iloc[:, list(idx)]
        DTE_filtered = DTE_filtered.iloc[:, list(idx)]


    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("pca", PCA(
            n_components=2,
            whiten=True,
        ))
    ])

    pipe.fit(DTR_filtered)
    DTR_pca = pipe.transform(DTR_filtered)
    DTE_pca = pipe.transform(DTE_filtered)

    sns.scatterplot(x=DTR_pca[:, 0], y=DTR_pca[:, 1], hue=LTR, alpha=0.7)
    sns.scatterplot(x=DTE_pca[:, 0], y=DTE_pca[:, 1], hue=LTE, marker="+", s=100, legend=False)
    plt.show()
    
printPCA_train_test(
    X_train, X_test, y_train, y_test,
    # regex="cc.rostrum_wavelet-HHL_LongRunLowGrayLevelEmphasis".replace("_", "_.*_"),
    regex="|".join(sfss["mlp"])
)
#%% notes

##### Logistic Regression |  C = 1
# _exponential_.*Median$

# _exponential_.*Mean$
# _wavelet-HLH_.*10Percentile$

# _wavelet-LLL_.*Minimum$
#----------------------------- | C = 1e-3
# _wavelet-HHH_.*HighGrayLevelEmphasis$
# _wavelet-HHH_.*SmallAreaHighGrayLevelEmphasis$

##### SVM
# _gradient_.*Kurtosis$

# _wavelet-HHL_.*DifferenceEntropy$
# _wavelet-HHL_.*DifferenceVariance$

##### MLP
# _wavelet-LLL_.*RootMeanSquared$
# _exponential_.*Median$
# _logarithm_.*Mean$
# _wavelet-HLH_.*10Percentile$
