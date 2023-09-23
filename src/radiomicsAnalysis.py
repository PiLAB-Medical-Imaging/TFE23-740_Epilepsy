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
import itertools

from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures, SmartCorrelatedSelection, SelectByInformationValue, SelectByShuffling
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, brier_score_loss, log_loss, balanced_accuracy_score, accuracy_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, LeaveOneOut, train_test_split, LeavePOut, StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.stats.multitest import fdrcorrection, multipletests
from statsmodels import robust
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from tqdm import tqdm
from tools_analysis import auc_and_f1

#%% Import dataset
# Import dataset

ds_path = "../study/stats/datasetRadiomics.csv"
df = pd.read_csv(ds_path, index_col="ID")
df = df.drop(["VNSLC_16"])
df = df.dropna(axis=1)
df.shape

#%% Save the reduced dataframe
# Save the reduced dataframe
X, y = df.iloc[:,2:], df.iloc[:,:1].squeeze()

df = df.drop(df.filter(regex=r'JointAverage').columns, axis=1)
df = df.drop(df.filter(regex=r'SumAverage').columns, axis=1) 
df = df.drop(df.filter(regex=r'SumSquares').columns, axis=1)
df = df.drop(df.filter(regex=r'_lbp-2D_').columns, axis=1)

pipe = Pipeline([
    ("constantFeatures", DropConstantFeatures(tol=0.62)),
    ("duplicateFeatures", DropDuplicateFeatures()),
])

df = pipe.fit_transform(df, y)

df.to_csv("../study/stats/datasetRadiomicsReduced.csv")

#%% Load reduced
df = pd.read_csv("../study/stats/datasetRadiomicsReduced.csv", index_col="ID", engine="c")
print(df.shape)

X, y = df.iloc[:,2:], df.iloc[:,:1].squeeze()

del df

#%% MAD outlier remotion
# MAD outlier remotion


def doubleMAD(X):
    m = X.median()
    abs_centred_median = abs(X-m)
    left = abs_centred_median[X<=m].median()
    right = abs_centred_median[X>=m].median()
    return left, right

def doubleMADsFromMedian(X):
    left, right = doubleMAD(X)
    m = X.median()
    mad = np.copy(np.broadcast_to(left, X.shape))
    right_full = np.copy(np.broadcast_to(right, X.shape))
    mad[X>m] = right_full[X>m]
    distance = abs(X-m)/mad
    distance[X==m] = 0
    return distance

class MADOutlierRemotion(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_val=3) -> None:
        self.threshold_val = threshold_val
        super().__init__()

    def fit(self, X, y=None):
        mask_outliers = doubleMADsFromMedian(X) > self.threshold_val

        self.features_mask = mask_outliers.any(axis=0)
        self.features_mask = self.features_mask == False
        return self

    def transform(self, X, y=None):
        return X.loc[:, self.features_mask]
    
    def get_support(self, indices=False):
        if indices:
            return np.argwhere(self.features_mask).ravel()
        return self.features_mask
    
    def get_limits(self):
        return self.limits

#%% Standard Scaler
# Standard Scaler

class MyScaler(RobustScaler):
    def transform(self, X, copy=None):
        if self.with_centering:
            X -= self.center_
        if self.with_scaling:
            X /= self.scale_
        return X

#%% Mannwhiten filtering (non-parametric of t-test)
# Mannwhiten filtering (non-parametric of t-test)

class MannwhitenFiltering(BaseEstimator, TransformerMixin):
    def __init__(self, pvalue_threshold=0.05) -> None:
        self.pvalue_threshold = pvalue_threshold
        super().__init__()
        
    def fit(self, X, y):
        # y is supposted to be binary {0, 1}

        X_true, X_false = X[y==1], X[y==0]
        pvalues = stats.mannwhitneyu(X_true, X_false).pvalue
        self.filtered_mask = pvalues < self.pvalue_threshold
        return self

    def transform(self, X, y=None):
        return X.loc[:, self.filtered_mask]

    def get_support(self, indices=False):
        if indices:
            return np.argwhere(self.filtered_mask).ravel()
        return self.filtered_mask

#%% Kurtosis
# Kurtosis

class KurtosisFiltering(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_min=-0.5, threshold_max=2) -> None:
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        super().__init__()
        
    def fit(self, X, y):
        # features_to_remove = []
        # for i in y.unique():
        #     X_curr = X[y==i]
        #    
        #     res = stats.kurtosis(X_curr, axis=0, bias=False)
        #     potential_outlier_features = (res<=self.threshold_min) | (res>=self.threshold_max)
        #     # potential_outlier_features = (res>=self.threshold_max)
        # 
        #     features_to_remove.append(potential_outlier_features)

        res = stats.kurtosis(X, axis=0, bias=False)
        features_potential_outliers_mask = (res<=self.threshold_min) | (res>=self.threshold_max)

        # features_to_remove = np.array(features_to_remove)
        # features_potential_outliers_mask = features_to_remove.any(axis=0)
        self.filtered_mask = (features_potential_outliers_mask == False)
        return self

    def transform(self, X, y=None):
        return X.loc[:, self.filtered_mask]

    def get_support(self, indices=False):
        if indices:
            return np.argwhere(self.filtered_mask).ravel()
        return self.filtered_mask

#%% Clustering plotting
# Clustering plotting

class saveCluster(BaseEstimator, TransformerMixin):
    def __init__(self, name) -> None:
        self.name = name
        super().__init__()

    def fit(self, X, y):
        if y.unique().size == 2:
            lut = dict(zip([0, 1, 2], "ryg"))
        elif y.unique().size == 3:
            lut = dict(zip([0, 1, 2], "ryg"))
        else:
            raise Exception()
        
        self.row_colors = y.map(lut)
        return self
    
    def transform(self, X, y=None):
        sns.clustermap(X, method="ward", row_colors=self.row_colors, z_score=1, robust=True, xticklabels=False)
        plt.savefig(f"../imgs/clustering/roi/{self.name}.png")
        return self

#%% RFE
# RFE

class MyRFECV(RFECV):

    def transform(self, X):
        self.columns = X.columns
        self.indices = X.index
        temp = super().transform(X)
        self.columns = self.columns[self.get_support()]
        return pd.DataFrame(temp, index=self.indices, columns=self.columns)

#%% Print features
# Print Features
class PrintFeatures(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y):
        return self

    def transform(self, X):
        if X.shape[0] == 1:
            print(X.shape)
        return X

#%% SFS
# SFS

class MySFS(SFS):
    def fit(self, X, y, groups=None, **fit_params):
        self.k_features = (1, X.shape[1])
        super().fit(X, y, groups, **fit_params)
        print("SFS", len(self.k_feature_idx_))
        if len(self.k_feature_idx_):
            print(X.columns[list(self.k_feature_idx_)])
        return self

    def transform(self, X):
        self.columns = X.columns
        self.indices = X.index
        temp = super().transform(X)
        self.columns = self.columns[list(self.k_feature_idx_)]
        return pd.DataFrame(temp, index=self.indices, columns=self.columns)

#%% Score estimation
# Score estimation

def retProbs(y_true, y_prob, **kwargs):
    # if y_true.size != 1:
    #     return roc_auc_score(y_true, y_prob)
    return y_prob

def printScores(y_true, y_prob, kind = None):

    def calc_f1(p_and_r):
        p, r = p_and_r
        if p == 0 and r == 0:
            return 0
        return (2*p*r)/(p+r)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob, pos_label=1)
    best_f1_index = np.argmax([ calc_f1(p_r) for p_r in zip(precisions, recalls)])
    best_threshold, best_precision, best_recall = thresholds[best_f1_index], precisions[best_f1_index], recalls[best_f1_index]

    y_pred = np.where(y_prob > best_threshold, 1, 0)

    roc = roc_auc_score(y_true, y_prob)
    if kind != "thresh":
        f1 = f1_score(y_true, y_pred)
        acc = balanced_accuracy_score(y_true, y_pred, adjusted=True)
        brier = brier_score_loss(y_true, y_prob)
        log = log_loss(y_true, y_prob)

    print("AUC:\t\tTest: %.3f" % (roc))
    if kind != "thresh":
        print("f1:\t\tTest: %.3f" % (f1))
        print("Accuracy:\tTest: %.3f" % (acc))
        print("Brier:\t\tTest: %.3f" % (brier))
        print("LogLoss:\tTest %.3f" % (log))

    return roc

#%% Finding the most characteristic images
# Finding the most characteristic images

for image_type in [
    "_exponential_", "_gradient_", 
    "_lbp-3D-k_", "_lbp-3D-m1_", "_lbp-3D-m2_",
    "_logarithm_", "_original_", "_square_", "_squareroot_",
    "_wavelet-HHH_", "_wavelet-HHL_", "_wavelet-HLH_", "_wavelet-HLL_",
    "_wavelet-LHH_", "_wavelet-LHL_", "_wavelet-LLH_", "_wavelet-LLL_",
]:
    print(image_type)
    cv = cross_validate(
        Pipeline([
            ("outliers", MADOutlierRemotion(3)),
            ("scaler", MyScaler()),
            ("rfe", MyRFECV(
                SVC(
                    C=1,
                    kernel="linear",
                    random_state=7,
                ),
                min_features_to_select=1,
                step=0.01,
                cv=StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=7
                ),
                scoring="roc_auc",
            )),
            # ("print", PrintFeatures()),
            ("clf", LogisticRegression(
                C=1e-3,
                random_state=7,
                dual=True, 
                solver="liblinear",
            )),
        ]),
        X.filter(regex=image_type), y,
        scoring=make_scorer(retProbs, needs_threshold=True),
        cv=LeaveOneOut(),
        n_jobs=4,
        verbose=5,
        error_score="raise",
    )

    printScores(y, cv["test_score"], "thresh")
    print()

#%% Finding the most characteristic images
# Finding the most characteristic images

loop = {
    "SVM [C=1e-3]":
        Pipeline([
            ("outliers", MADOutlierRemotion(3)),
            ("scaler", MyScaler()),
            ("clf", SVC(
                C=1e-3,
                random_state=7
            )),
        ])
    ,
    "SVM [C=1e-3] + RFE(LogReg [C=1])": 
        Pipeline([
            ("outliers", MADOutlierRemotion(3)),
            ("scaler", MyScaler()),
            ("rfe", MyRFECV(
                LogisticRegression(
                    random_state=7,
                ),
                min_features_to_select=1,
                step=0.01,
                cv=StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=7
                ),
                scoring="roc_auc",
            )),
            ("clf", SVC(
                C=1e-3,
                random_state=7
            )),
        ])    
    ,
    "SVM [C=1e-3] + RFE(LogReg [C=1e-3])" : 
        Pipeline([
            ("outliers", MADOutlierRemotion(3)),
            ("scaler", MyScaler()),
            ("rfe", MyRFECV(
                LogisticRegression(
                    C=1e-3,
                    random_state=7,
                ),
                min_features_to_select=1,
                step=0.01,
                cv=StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=7
                ),
                scoring="roc_auc",
            )),
            ("clf", SVC(
                C=1e-3,
                random_state=7
            )),
        ])
    ,
    "SVM [C=1e-3] + RFE(SVM    [C=1e-3])" : 
        Pipeline([
            ("outliers", MADOutlierRemotion(3)),
            ("scaler", MyScaler()),
            ("rfe", MyRFECV(
                SVC(
                    C=1e-3,
                    kernel="linear",
                    random_state=7,
                ),
                min_features_to_select=1,
                step=0.01,
                cv=StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=7
                ),
                scoring="roc_auc",
            )),
            ("clf", SVC(
                C=1e-3,
                random_state=7
            )),
        ])
    ,
    "SVM [C=1e-3] + RFE(SVM    [C=1])" : 
        Pipeline([
            ("outliers", MADOutlierRemotion(3)),
            ("scaler", MyScaler()),
            ("rfe", MyRFECV(
                SVC(
                    kernel="linear",
                    random_state=7,
                ),
                min_features_to_select=1,
                step=0.01,
                cv=StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=7
                ),
                scoring="roc_auc",
            )),
            ("clf", SVC(
                C=1e-3,
                random_state=7
            )),
        ])
    ,
    "LogReg[C=1e-3]" : 
        Pipeline([
            ("outliers", MADOutlierRemotion(3)),
            ("scaler", MyScaler()),
            ("clf", LogisticRegression(
                dual=True,
                solver="liblinear",
                C=1e-3,
                random_state=7
            )),
        ])
    ,
    "LogReg[C=1e-3] + RFE(LogReg [C=1])" : 
        Pipeline([
            ("outliers", MADOutlierRemotion(3)),
            ("scaler", MyScaler()),
            ("rfe", MyRFECV(
                LogisticRegression(
                    random_state=7,
                ),
                min_features_to_select=1,
                step=0.01,
                cv=StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=7
                ),
                scoring="roc_auc",
            )),
            ("clf", LogisticRegression(
                dual=True,
                solver="liblinear",
                C=1e-3,
                random_state=7
            )),
        ])
    ,
    "LogReg[C=1e-3] + RFE(LogReg [C=1e-3])" : 
        Pipeline([
            ("outliers", MADOutlierRemotion(3)),
            ("scaler", MyScaler()),
            ("rfe", MyRFECV(
                LogisticRegression(
                    C=1e-3,
                    random_state=7,
                ),
                min_features_to_select=1,
                step=0.01,
                cv=StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=7
                ),
                scoring="roc_auc",
            )),
            ("clf", LogisticRegression(
                dual=True,
                solver="liblinear",
                C=1e-3,
                random_state=7
            )),
        ])
    ,
    "LogReg[C=1e-3] + RFE(SVM    [C=1e-3])" : 
        Pipeline([
            ("outliers", MADOutlierRemotion(3)),
            ("scaler", MyScaler()),
            ("rfe", MyRFECV(
                SVC(
                    C=1e-3,
                    kernel="linear",
                    random_state=7,
                ),
                min_features_to_select=1,
                step=0.01,
                cv=StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=7
                ),
                scoring="roc_auc",
            )),
            ("clf", LogisticRegression(
                dual=True,
                solver="liblinear",
                C=1e-3,
                random_state=7
            )),
        ])
    ,
    "LogReg[C=1e-3] + RFE(SVM    [C=1])" : 
        Pipeline([
            ("outliers", MADOutlierRemotion(3)),
            ("scaler", MyScaler()),
            ("rfe", MyRFECV(
                SVC(
                    kernel="linear",
                    random_state=7,
                ),
                min_features_to_select=1,
                step=0.01,
                cv=StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=7
                ),
                scoring="roc_auc",
            )),
            ("clf", LogisticRegression(
                dual=True,
                solver="liblinear",
                C=1e-3,
                random_state=7
            )),
        ])
    ,
}

#%%
res = {}
for name, pipe in loop.items():
    res[name] = {}
    for image_type in [
        "_exponential_", "_gradient_", 
        "_lbp-3D-k_", "_lbp-3D-m1_", "_lbp-3D-m2_",
        "_logarithm_", "_original_", "_square_", "_squareroot_",
        "_wavelet-HHH_", "_wavelet-HHL_", "_wavelet-HLH_", "_wavelet-HLL_",
        "_wavelet-LHH_", "_wavelet-LHL_", "_wavelet-LLH_", "_wavelet-LLL_",
    ]:
        res[name][image_type] = {}
        for feature_type in [
            "firstorder_", "glcm_", "gldm_", "glrlm_", "glszm_", "ngtdm_"
            "shape_"
        ]:
            X_curr = X.filter(regex=image_type+feature_type)
            print(name)
            print(X_curr.shape)
            if X_curr.shape[1] == 0:
                continue
            
            print(image_type+feature_type)
            cv = cross_validate(
                pipe,
                X_curr, y,
                scoring=make_scorer(retProbs, needs_threshold=True),
                cv=LeaveOneOut(),
                n_jobs=4,
                # verbose=5,
                error_score="raise",
            )

            res[name][image_type][feature_type] = printScores(y, cv["test_score"], "thresh")
            print()

with open("../study/stats/results-all.json", "w") as outfile:
    json.dump(res, outfile, indent=2, sort_keys=True)
#%% Statistics
# Statistics

with open('../study/stats/results-all.json') as infile:
    res = json.load(infile)

avg_accuracy = {}

counters = {}

for model in res.keys():
    avg_accuracy[model] = 0
    n = 0
    for image_type in res[model].keys():
        for feature_type in res[model][image_type].keys():
            avg_accuracy[model] += res[model][image_type][feature_type]
            n += 1

            lev = int(res[model][image_type][feature_type]*10)*100

            if lev < 700:
                continue

            if image_type+feature_type not in counters:
                counters[image_type+feature_type] = {}
            
            if lev not in counters[image_type+feature_type]:
                counters[image_type+feature_type][lev] = 0

            counters[image_type+feature_type][lev] += 1

    avg_accuracy[model] /= n

counters

#%% Statistics 
# Statistics

selected_groups = []

for sub_samples in counters.keys():
    count_over = 0
    for level in counters[sub_samples].keys():
        if level < 700:
            continue
        count_over += counters[sub_samples][level]

    if count_over >=4:
        selected_groups.append(sub_samples)

selected_groups

#%% Combinatorial selection
# Combinatorial selection

selected_groups = np.array(selected_groups)
mask = np.zeros(selected_groups.shape).astype(bool)

res = {}

def run_cv(X, y, selected_groups):
    for name, pipe in loop.items():
        print(name)
        print(X.shape)

        print(selected_groups)
        cv = cross_validate(
            pipe,
            X, y,
            scoring=make_scorer(retProbs, needs_threshold=True),
            cv=LeaveOneOut(),
            n_jobs=4,
            # verbose=5,
            error_score="raise",
        )

        res[selected_groups][name] = printScores(y, cv["test_score"], "thresh")
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
        if v.sum() <= 1: 
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

with open("../study/stats/results-comb.json", "w") as outfile:
    json.dump(res, outfile, indent=2, sort_keys=True)

#%% Statistics 
# Statistics

with open('../study/stats/results-comb.json') as infile:
    res = json.load(infile)

counts = {}
avg_model_scores = {}

for image_feature_group in res.keys():
    counts[image_feature_group] = {}
    for model in res[image_feature_group].keys():

        if model not in avg_model_scores:
            avg_model_scores[model] = 0
        avg_model_scores[model] += res[image_feature_group][model]
        
        level = int(res[image_feature_group][model]*10)*100

        if level < 800:
            continue
        
        if level not in counts[image_feature_group]:
            counts[image_feature_group][level] = []
        
        counts[image_feature_group][level].append(round(res[image_feature_group][model], 3))
    
    if len(counts[image_feature_group]) == 0:
        del counts[image_feature_group]
    
for score_model in avg_model_scores.keys():
    avg_model_scores[score_model] /= len(res)

avg_model_scores

#%% Statistics
# Statistics

selected_groups = []

for feature_group, vals in counts.items():
    if len(vals) >=2:
        selected_groups.append(feature_group)

selected_groups

#%% Randomized Search Naive Bayes
# Randomized Search

n_iter = 10

begin_pipe = {
    "outliers" : MADOutlierRemotion(3),
    "scaler" : MyScaler(),
}

outer_loop = {
    "SVM [linear]" : {
        "clf" : SVC(
                    kernel="linear",
                    random_state=7,
                ),
        "hyperparams" : {
            "clf__C" : stats.loguniform(1e-6, 1e-0)
        },
        "n_iter" : n_iter
    },
    "SVM [poly]" : {
        "clf" : SVC(
                    kernel="poly",
                    random_state=7,
                ),
        "hyperparams" : {
            "clf__C" : stats.loguniform(1e-6, 1e-0),
            "clf__degree" : [2, 3, 4]
        },
        "n_iter" : n_iter
    },
    "SVM [rbf]" : {
        "clf" : SVC(
                    kernel="rbf",
                    random_state=7,
                ),
        "hyperparams" : {
            "clf__C" : stats.loguniform(1e-6, 1e-0),
            "clf__gamma" : ["scale", "auto"]
        },
        "n_iter" : n_iter
    },
    "LogReg" : {
        "clf" : LogisticRegression(
            dual=True,
            solver="liblinear",
            random_state=7,
        ),
        "hyperparams": {
            "clf__C" : stats.loguniform(1e-6, 1e-0),
        },
        "n_iter" : n_iter
    },
    "GaussianNB" : {
        "clf" : GaussianNB(),
        "hyperparams" : {},
        "n_iter" : 1,
    },
    "K Nearest Neighbors" : {
        "clf" : KNeighborsClassifier(),
        "hyperparams" : {
            "clf__n_neighbors": [2, 3, 4, 5, 6],
        },
        "n_iter" : n_iter,
    },
    "Radius Nearest Neighbors" : {
        "clf" : RadiusNeighborsClassifier(
            outlier_label="most_frequent",
        ),
        "hyperparams" : {
            "clf__radius": stats.uniform(0, 2),
        },
        "n_iter" : n_iter,
    },
    "Random Forest" : {
        "clf" : RandomForestClassifier(
            random_state=7
        ),
        "hyperparams" : {
            "clf__n_estimators" : stats.randint(100, 500),
            "clf__criterion" : ["gini", "entropy", "log_loss"],
            "clf__max_features" : stats.randint(2, 10),
            "clf__max_leaf_nodes" : stats.randint(4, 10),
            "clf__ccp_alpha" : stats.uniform(0, 1),
        },
        "n_iter" : n_iter,
    },
}

inner_loop = {
    "none" : None,
    "RFE [SVM]" : {
        "rfe" : MyRFECV(
            SVC(
                kernel="linear",
                random_state=7,
            ),
            min_features_to_select=2,
            step=0.01,
            cv=StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=7
            ),
            scoring="roc_auc"
        ),
        "hyperparams" : {
            "rfe__estimator__C" : stats.loguniform(1e-6, 1e-0)
        }
    },
    "RFE [LogReg]" : {
        "rfe" : MyRFECV(
            LogisticRegression(
                random_state=7,
            ),
            min_features_to_select=2,
            step=0.01,
            cv=StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=7
            ),
            scoring="roc_auc"
        ),
        "hyperparams" : {
            "rfe__estimator__C" : stats.loguniform(1e-6, 1e-0)
        }
    },
}

inner2_loop = {
    "none" : None,
    "PCA" : {
        "pca" : PCA(
            n_components=2,
            whiten=True,
            random_state=7
        ),
        "hyperparams" : {
            "pca__n_components" : stats.randint(3, 20)
        }
    }
}

res = {}
for classifier_name in outer_loop.keys():
    clf_params = outer_loop[classifier_name]
    res[classifier_name] = {}
    for selection_name in inner_loop.keys():
        selct_params = inner_loop[selection_name]
        res[classifier_name][selection_name] = {}
        for reduction_name in inner2_loop.keys():
            reduction_params = inner2_loop[reduction_name]

            full_pipe = dict(begin_pipe)
            param_distribution = dict(clf_params["hyperparams"])

            if selct_params is not None:
                full_pipe["rfe"] = selct_params["rfe"]
                param_distribution_selection = selct_params["hyperparams"]
                param_distribution = dict(param_distribution, **param_distribution_selection)

            if reduction_params is not None:
                full_pipe["pca"] = reduction_params["pca"]

            full_pipe["clf"] = clf_params["clf"]

            # print(full_pipe)
            print(classifier_name, selection_name, reduction_name)

            full_pipe = Pipeline(list(full_pipe.items()))

            cv = cross_validate(
                RandomizedSearchCV(
                    full_pipe,
                    param_distributions=param_distribution,
                    n_iter=clf_params["n_iter"],
                    scoring=("roc_auc"),
                    refit="roc_auc",
                    cv=StratifiedKFold(
                        n_splits=3,
                        shuffle=True,
                        random_state=7
                    ),
                    random_state=7,
                    error_score="raise",
                    return_train_score=True,
                ),
                X.filter(regex=r'_logarithm_firstorder_|_wavelet-LLH_glcm_'), y,
                cv=LeaveOneOut(),
                scoring=make_scorer(retProbs, needs_threshold=True),
                n_jobs=-1,
                verbose=10,
                return_estimator=True,
                error_score="raise"
            )

            res[classifier_name][selection_name][reduction_name] = (cv, printScores(y, cv["test_score"], "thresh"))
