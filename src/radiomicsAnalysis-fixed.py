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
from sklearn.preprocessing import RobustScaler
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
X_train, X_test, y_train, y_test = utils.getTrainTestSplits()

#%% best feature groups computation
# best feature groups

for name, algorithm in [
    ("logreg", LogisticRegression(C=1e-6, dual=True, solver="liblinear", random_state=7, max_iter=1000, class_weight="balanced")),
    ("svm", SVC(C=1e-6, random_state=7, class_weight="balanced")),
    ("knn", CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=7))),
    ("mlp", MLPClassifier(random_state=7, alpha=2, learning_rate="adaptive", max_iter=500)),
    ("gauss", GaussianNB())
]:

    res = {}

    feature_names = utils.getPyRadiomicsFeatureNames()
    image_types = utils.getPyRadiomicsImageTypes()
    region_names = utils.getRegionNames()

    for region in tqdm(region_names):
        for image_type in image_types:
            for feature_name in feature_names:

                print(region+"_"+image_type+"_"+feature_name)

                res[region+"_"+image_type+"_"+feature_name] = utils.scoreLOO(
                    algorithm,
                    X_train, y_train,
                    regex = region+"_.*_"+image_type+"_.*_"+feature_name,
                    decision=True if name == "svm" else False
                )

    with open(f"../study/stats/results-{name}-fix40.json", "w") as outfile:
        json.dump(res, outfile, indent=2, sort_keys=True)

#%% selection

best_selected_features = {}

for name in ["logreg", "svm"]:
    res = {}
    with open(f"../study/stats/results-{name}-fix40.json") as infile:
        res = json.load(infile)
    res_df = pd.DataFrame(res).T
    best_image_features = res_df[res_df["acc"]>0.8].dropna().sort_values(by="acc", ascending=False).iloc[:10, :].index
    regex = "|".join(best_image_features)

    best_selected_features[name] = regex


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

sfss = {}

for name, algorithm, selected_groups in [
    (
        "logreg",
        LogisticRegression(C=1e-6, dual=True, solver="liblinear", random_state=7, max_iter=100000, class_weight="balanced"),
        "_original_.*Range$"
    ),
    # (
    #     "svm",
    #     CalibratedClassifierCV(SVC(C=1e-3, random_state=7, class_weight="balanced"), cv=3),
    #     "_gradient_.*Kurtosis$|_wavelet-HHL_.*DifferenceEntropy$|_wavelet-HHL_.*DifferenceVariance$"
    # ),
    # # ("knn", CalibratedClassifierCV(KNeighborsClassifier(), cv=3)),
    # (
    #     "mlp", 
    #     MLPClassifier(random_state=7, alpha=2, learning_rate="adaptive", max_iter=1000,),
    #     "_wavelet-LLL_.*RootMeanSquared$|_exponential_.*Median$|_logarithm_.*Mean$|_wavelet-HLH_.*10Percentile$"
    # ),
    # # ("gauss", GaussianNB())
]:
    
    X_filtered = X_train.filter(regex=selected_groups)

    sfs = MySFS(
        Pipeline([
            ("scaler", MyScaler()),
            ("clf", algorithm)
        ]),
        k_features=(1,5),
        floating=True,
        scoring="balanced_accuracy",
        cv=3,
        n_jobs=9,
        verbose=10,
    )

    sfs.fit(X_filtered, y_train)

    sfss[name] = sfs
#%% plot sfs
plot_sfs(sfss["logreg"].get_metric_dict(), kind='std_dev')
#%% Best Log Reg

fitTrain_scoreTest(
    SVC(C=1e-6, random_state=7, class_weight="balanced"),
    X_train, X_test, y_train, y_test,
    "_original_.*Range$",
    # sfss["logreg"].get_metric_dict()[8]["feature_idx"]
    sfss["logreg"].k_feature_idx_
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

