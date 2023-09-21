#####
# Classification of Responders and Non-responders with 
#####

#%% Import
# Import
from typing import Any
from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import fastcluster
import json
import itertools

from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, brier_score_loss, log_loss, balanced_accuracy_score, accuracy_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, LeaveOneOut, train_test_split, LeavePOut, StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.stats.multitest import fdrcorrection, multipletests
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

#%% Remove duplicates
# Remove duplicates

class DropDuplicatesColumns(BaseEstimator, TransformerMixin):
    def __init__(self, keep="fist") -> None:
        super().__init__()
        self.keep = keep

    def fit(self, X, y=None):
        self.duplicates_mask = X.T.duplicated()
        self.duplicates_mask = (self.duplicates_mask == False)
        return self

    def transform(self, X, y=None):
        return X.loc[:, self.duplicates_mask]
    
    def get_support(self, indices=False):
        if indices:
            return np.argwhere(self.duplicates_mask).ravel()
        return self.duplicates_mask

#%% Variance threshold
# Variance threshold

class MyVarianceThreshold(VarianceThreshold):
    def __init__(self, threshold=0) -> None:
        super().__init__(threshold)

    def fit(self, X, y=None):
        return super().fit(X, y)
    
    def transform(self, X):
        return X.loc[:, self.get_support()]

#%% Save the reduced dataframe
varThresh = MyVarianceThreshold()
df = df.T.drop_duplicates().T
df = varThresh.fit_transform(df)

df.to_csv("../study/stats/datasetRadiomicsReduced.csv")
#%% Load reduced
df = pd.read_csv("../study/stats/datasetRadiomicsReduced.csv", index_col="ID")
df.shape

X, y = df.iloc[:,2:], df.iloc[:,:1].squeeze()
del df
    
#%% Standard Scaler
# Standard Scaler

class MyStandardScaler(StandardScaler):
    def transform(self, X, copy=None):
        if self.with_mean:
            X -= self.mean_
        if self.with_std:
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

#%% Groups outlier remotion
# Groups outlier remotion

class GroupsOutlierRemotion(BaseEstimator, TransformerMixin):
    def __init__(self, z_threshold=3) -> None:
        self.z_threshold = z_threshold
        super().__init__()

    def fit(self, X, y):
        features_to_remove = []
        for i in y.unique():
            X_curr = abs(X[y==i])

            features_outliers_mask = (X_curr > self.z_threshold).any(axis=0)

            features_to_remove.append(features_outliers_mask)
        
        features_to_remove = np.array(features_to_remove)
        features_outliers_mask = features_to_remove.any(axis=0)
        self.features_outliers_mask = (features_outliers_mask == False)
        return self

    def transform(self, X, y=None):
        return X.loc[:, self.features_outliers_mask]
    
    def get_support(self, indices=False):
        if indices:
            return np.argwhere(self.features_outliers_mask).ravel()
        return self.features_outliers_mask

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

#%% Redudant (Correlation)
# Redudant (Correlation)
class DropRedundantColumns(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_corr=0.99) -> None:
        self.threshold_corr = threshold_corr
        super().__init__()

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        corr_matrix = X_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype("bool"))
        to_drop = [True if any(upper[column]>=self.threshold_corr) else False for column in upper.columns]
        self.filtered_mask = np.array(to_drop) == False
        print("Correlation", self.filtered_mask[self.filtered_mask].size)
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
    def fit(self, X, y, groups=None):
        super().fit(X, y, groups)
        print("RFE", self.get_support()[self.get_support() == True].size)
        return self

    def transform(self, X):
        self.columns = X.columns
        self.indices = X.index
        temp = super().transform(X)
        self.columns = self.columns[self.get_support()]
        return pd.DataFrame(temp, index=self.indices, columns=self.columns)

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

def printScores(y_true, y_prob):

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
    f1 = f1_score(y_true, y_pred)
    acc = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    brier = brier_score_loss(y_true, y_prob)
    log = log_loss(y_true, y_prob)

    print("AUC:\t\tTest: %.3f" % (roc))
    print("f1:\t\tTest: %.3f" % (f1))
    print("Accuracy:\tTest: %.3f" % (acc))
    print("Brier:\t\tTest: %.3f" % (brier))
    print("LogLoss:\tTest %.3f" % (log))

#%% Randomized Search Naive Bayes
# Randomized Search

distrib = dict([
    ("clf__C", stats.loguniform(1e-6, 1e-0))
])

cv = cross_validate(
    RandomizedSearchCV(
        Pipeline([
            ("duplicates", DropDuplicatesColumns()),
            ("variance", MyVarianceThreshold()),
            ("scaler", MyStandardScaler()),
            ("outliers", GroupsOutlierRemotion(2)),
            ("kurtosis", KurtosisFiltering(-0.5, 1)),
            ("mannwhiten", MannwhitenFiltering(0.05)),
            # No Kruskal since we do only the binary case
            ("redundant", DropRedundantColumns(0.95)),
            ("rfe", MyRFECV(
                LogisticRegression(),
                min_features_to_select=1,
                step=0.01,
                cv=StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=7
                ),
                scoring=make_scorer(auc_and_f1, needs_proba=True),
            )),
            ("clf", LogisticRegression(
                dual=True,
                solver="liblinear"
            ))
        ]),
        param_distributions=distrib,
        n_iter=1,
        scoring=("balanced_accuracy", "f1", "roc_auc", "neg_brier_score", "neg_log_loss"),
        refit="roc_auc", # to change
        cv=StratifiedKFold(
                n_splits=6,
                shuffle=True,
                random_state=7
            ),
        random_state=7,
        error_score="raise",
        return_train_score=True,
    ),
    X, y,
    scoring=make_scorer(retProbs, needs_proba=True),
    cv=LeaveOneOut(),
    n_jobs=6,
    verbose=10,
    return_estimator=True,
    error_score="raise"
)

printScores(y, cv["test_score"])