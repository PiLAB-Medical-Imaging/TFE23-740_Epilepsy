import json
import utils
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from tqdm import tqdm
from numpy import ndarray

def manageDefault(element):
    if type(element) == ndarray :
        return list(element)
    raise Exception

def getFeatureGroups(X_only_pyRadiomicsFeatures):
    X = X_only_pyRadiomicsFeatures
    features = pd.Series(X.columns)

    regions = features.map(lambda x: x.split("_")[0])
    pyradiomicImages = features.map(lambda x: x.split("_")[-3])
    pyradiomicFeatures = features.map(lambda x: x.split("_")[-1])
    
    multiIndices = pd.MultiIndex.from_arrays((regions, pyradiomicImages, pyradiomicFeatures), names=("Region", "Image", "Feature"))
    
    base = pd.Series(np.arange(regions.size), index=multiIndices, name="FeatureGroups")
    
    return base.groupby(level=["Region", "Image", "Feature"]).aggregate(lambda x: [*x]).to_list()

df = utils.getReducedDS()
X, y, y3 = utils.splitFeatureLabels(df)
X_train, X_test, y_train, y_test, y3_train, y3_test = utils.splitTrainTestDF(X, y, y3, 0.4)

for name, algorithm in [
    ("logreg", LogisticRegression(C=1e-6, random_state=7, max_iter=1000, class_weight="balanced")),
    ("svm", SVC(C=1e-6, random_state=7, class_weight="balanced")),
    ("knn", CalibratedClassifierCV(KNeighborsClassifier(), cv=3)),
    ("mlp", MLPClassifier(random_state=7, alpha=2, learning_rate="adaptive", max_iter=500)),
    ("gauss", GaussianNB())
]:
    
    pipe = Pipeline([
        ("outliers", utils.MADOutlierRemotion(3)),
        ("scaler", RobustScaler()),
        ("clf", algorithm)
    ])

    sfs = SFS(
        pipe,
        k_features=1,
        forward=True,
        floating=True,
        scoring="roc_auc",
        cv=StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=7,
        ),
        verbose=10,
        n_jobs=8,
        feature_groups=getFeatureGroups(X_train.iloc[:, 6:])
    )

    sfs.fit(X_train.iloc[:, 6:], y_train)

    with open(f"../study/stats/results-{name}-fix40-sfs-3cv.json", "w") as outfile:
        json.dump(sfs.get_metric_dict(), outfile, indent=2, sort_keys=True, default=manageDefault)