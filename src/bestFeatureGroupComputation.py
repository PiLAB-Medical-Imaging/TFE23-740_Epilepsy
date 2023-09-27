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

X_train, X_test, y_train, y_test = utils.getTrainTestSplits()

for name, algorithm in [
    ("logreg", LogisticRegression(C=1e-6, dual=True, solver="liblinear", random_state=7, max_iter=1000, class_weight="balanced")),
    ("svm", SVC(C=1e-6, random_state=7, class_weight="balanced")),
    ("knn", CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=7))),
    ("mlp", MLPClassifier(random_state=7, alpha=2, learning_rate="adaptive", max_iter=500)),
    ("gauss", GaussianNB())
]:

    res = {}

    multiIndices = utils.countByFeatureGroups(X_train.iloc[:, 5:]).index

    for region, image_type, feature_name in tqdm(multiIndices):

        res[region+"_"+image_type+"_"+feature_name] = utils.scoreLOO(
            algorithm,
            X_train, y_train,
            regex = region+"_.*_"+image_type+"_.*_"+feature_name,
            decision=True if name == "svm" else False,
            doPrints=False
        )

    with open(f"../study/stats/results-{name}-fix40.json", "w") as outfile:
        json.dump(res, outfile, indent=2, sort_keys=True)