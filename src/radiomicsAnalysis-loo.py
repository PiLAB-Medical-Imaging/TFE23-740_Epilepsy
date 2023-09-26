#####
# Classification of Responders and Non-responders with 
#####

#%% Import
# Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import fastcluster
import json
import utils

from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, LeaveOneOut, train_test_split, RandomizedSearchCV, GridSearchCV

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

#%% Init

df = utils.getReducedDS()
X, y = utils.splitFeatureLabels(df)

#%% Combinatorial selection
# Combinatorial selection

selected_groups = ["_wavelet-LLH_glcm_", "_original_firstorder_","_squareroot_firstorder_", "_logarithm_firstorder_", "_wavelet-LLL_firstorder_"]

selected_groups = ["_original_firstorder_", "_logarithm_firstorder_"]

# selected_groups = ["_original_firstorder_", "_wavelet-LLL_firstorder_"]

selected_groups = np.array(selected_groups)
mask = np.zeros(selected_groups.shape).astype(bool)

pipe = Pipeline([
    ("outliers", MADOutlierRemotion(3)),
    ("scaler", MyScaler()),
    ("clf", CalibratedClassifierCV(KNeighborsClassifier()))
])

res = {}

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
        error_score="raise",
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
        
        print(selected_groups[mask])
        selected_groups_join = ";".join(selected_groups[mask])
        res[selected_groups_join] = {}
        run_cv(extract_df(X, selected_groups[mask]), y, selected_groups_join)
        return

    v[i] = True
    f(v, i+1)
    v[i] = False
    f(v, i+1)
    return

f(mask, 0)

with open("../study/stats/results-mlp-comb.json", "w") as outfile:
    json.dump(res, outfile, indent=2, sort_keys=True)
    
#%% best feature groups computation
#  best feature groups

for name, algorithm in [
    ("logreg", LogisticRegression(random_state=7)),
    ("svm", CalibratedClassifierCV(SVC(random_state=7))),
    ("knn", CalibratedClassifierCV(KNeighborsClassifier())),
    ("mlp", MLPClassifier(random_state=7)),
    ("gauss", GaussianNB())
]:

    res = {}

    for image_type in [
            "_exponential_", "_gradient_", 
            "_lbp-3D-k_", "_lbp-3D-m1_", "_lbp-3D-m2_",
            "_logarithm_", "_original_", "_square_", "_squareroot_",
            "_wavelet-HHH_", "_wavelet-HHL_", "_wavelet-HLH_", "_wavelet-HLL_",
            "_wavelet-LHH_", "_wavelet-LHL_", "_wavelet-LLH_", "_wavelet-LLL_",
        ]:
            for feature_type in distinct_features:

                pipe = Pipeline([
                    ("scaler", MyScaler()),
                    ("clf", algorithm)
                ])

                print(image_type+feature_type)
                X_filtered = X.filter(regex=image_type+feature_type)
                if X_filtered.shape[1] == 0:
                    continue

                cv = cross_validate(
                    pipe,
                    X_filtered,y,
                    scoring=make_scorer(retProbs, needs_proba=True),
                    cv=LeaveOneOut(),
                    n_jobs=-1,
                    verbose=1,
                    return_estimator=True,
                    error_score=0.5
                )

                res[image_type+feature_type] = printScores(y, cv["test_score"])

    with open(f"../study/stats/results-{name}.json", "w") as outfile:
        json.dump(res, outfile, indent=2, sort_keys=True)

#%% LogReg Best

res = utils.scoreLOO(
    LogisticRegression(), 
    X, y, 
    "_wavelet-LLH_.*_Contrast$|_lbp-3D-k_.*_SizeZoneNonUniformity$",
    confusion=True
)
    

#%%
distribution = {
    "clf__C": stats.loguniform(1e-6, 1e0)
}

search = RandomizedSearchCV(
    Pipeline([
        ("filter", utils.FilterDF("_lbp-3D-k_.*_SizeZoneNonUniformity$")),
        ("scaler", utils.RobustScalerDF()),
        ("clf", LogisticRegression())
    ]),
    param_distributions=distribution,
    n_iter=50,
    scoring="neg_brier_score",
    n_jobs=-1,
    cv=2,
    verbose=10
)

search.fit(X, y)

#%%

sfs = utils.SFSDF(
    Pipeline([
        ("scaler", utils.RobustScalerDF()),
        ("clf", LogisticRegression())
    ]),
    k_features=(1,20),
    floating=True,
    verbose=10,
    scoring="neg_brier_score",
    cv=3,
    n_jobs=-1
)

sfs.fit(X.filter(regex="_wavelet-LLH_.*_Contrast$|_lbp-3D-k_.*_SizeZoneNonUniformity$"), y)
#%%

plot_sfs(sfs.get_metric_dict(), kind='std_dev')

#%%

res = utils.scoreLOO(
    LogisticRegression(), 
    X, y, 
    "_wavelet-LLH_.*_Contrast$|_lbp-3D-k_.*_SizeZoneNonUniformity$",
    sfs.k_feature_idx_,
    confusion=True
)

#%% MLP Best 

res = utils.scoreLOO(
    MLPClassifier(),
    X, y,
    "_wavelet-LLH_.*Complexity$|_wavelet-LLH_.*Idmn$"
)

#%% KNN best

res = utils.scoreLOO(
    CalibratedClassifierCV(KNeighborsClassifier()),
    X, y,
    "_wavelet-LLH_.*Contrast$",
    confusion=True
)

#%%
distribution = {
    "clf__estimator__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

search = GridSearchCV(
    Pipeline([
        ("filter", utils.FilterDF("_wavelet-LLH_.*Contrast$")),
        ("scaler", utils.RobustScalerDF()),
        ("clf", CalibratedClassifierCV(KNeighborsClassifier()))
    ]),
    param_grid=distribution,
    scoring="neg_brier_score",
    n_jobs=-1,
    cv=3,
    verbose=10
)

search.fit(X, y)

#%% SVM best

res = utils.scoreLOO(
    CalibratedClassifierCV(SVC(C=0.003)),
    X, y,
    "_wavelet-LHH_.*JointEnergy$|_wavelet-LHH_.*MaximumProbability$",
    confusion=True
)
#%%
distribution = {
    "clf__estimator__C": stats.loguniform(1e-6, 1e0)
}

search = RandomizedSearchCV(
    Pipeline([
        ("filter", utils.FilterDF("_wavelet-LHH_.*JointEnergy$|_wavelet-LHH_.*MaximumProbability$")),
        ("scaler", utils.RobustScalerDF()),
        ("clf", CalibratedClassifierCV(SVC()))
    ]),
    param_distributions=distribution,
    n_iter=50,
    scoring="neg_brier_score",
    n_jobs=-1,
    cv=3,
    verbose=10
)

search.fit(X, y)
