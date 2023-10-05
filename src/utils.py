#### Utils ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import nilearn as nl

from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures, SmartCorrelatedSelection

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_validate, LeaveOneOut, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, brier_score_loss, log_loss, balanced_accuracy_score, accuracy_score, precision_recall_curve, confusion_matrix
from sklearn.ensemble import VotingClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from scipy import stats

#%% DATASET %%#

def getPathFullDS():
    return "../study/stats/datasetRadiomics.csv"

def getFullDS():
    df = pd.read_csv(getPathFullDS(), index_col="ID")
    df = df.drop(["VNSLC_16"])
    df = df.dropna(axis=1)
    return

def reduceSaveAndGetDS(df):
    feature_to_remove = {
        'JointAverage',
        'SumAverage',
        'SumSquares',
        '_lbp-2D_',
        'Minimum',
        'Maximum',
        '10Percentile',
        '90Percentile',
        'InterquartileRange',
        'Range',
        'Maximum2DDiameterColumn',
        'Maximum2DDiameterRow',
        'Maximum2DDiameterSlice',
        'Maximum3DDiameter',
    }

    for feature in feature_to_remove:
        df = df.drop(df.filter(regex=feature).columns, axis=1)
    
    pipe = Pipeline([
        ("constantFeatures", DropConstantFeatures(tol=0.62)),
        ("duplicateFeatures", DropDuplicateFeatures()),
    ])

    df = pipe.fit_transform(df)

    df.to_csv("../study/stats/datasetRadiomicsReduced.csv")

    return df

def getReducedDS():
    return pd.read_csv("../study/stats/datasetRadiomicsReduced.csv", index_col="ID")

def splitFeatureLabels(df: pd.DataFrame):
    return df.iloc[:,2:], df.iloc[:,:1].squeeze(), df.iloc[:,1:2].squeeze()

def __getACharacteristcByPos(pos):
    with open("../study/subjects/VNSLC_02/dMRI/microstructure/VNSLC_02_metrics.json") as infile:
        file = json.load(infile)

    distinct_features = set()

    for name in file.keys():
        if name == "ID":
            continue

        feature_name = name.split("_")[pos]
        distinct_features.add(feature_name)
    
    return distinct_features

def getPyRadiomicsFeatureNames():
    return __getACharacteristcByPos(-1)

def getRegionNames():
    return __getACharacteristcByPos(0)

def getPyRadiomicsImageTypes():
    t = __getACharacteristcByPos(-3)
    t.remove("csf")
    t.remove("tot")
    return t

def splitTrainTestDF(X, y, y3, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7, shuffle=True, stratify=y3)
    y3_train = y3[y_train.index]
    y3_test = y3[y_test.index]

    return X_train, X_test, y_train, y_test, y3_train, y3_test

def saveTrainTestSplits(X_train, X_test, y_train, y_test):
    pd.concat([y_train, X_train], axis=1).to_csv("../study/stats/datasetRadiomicsReducedTrain20.csv")
    pd.concat([y_test, X_test], axis=1).to_csv("../study/stats/datasetRadiomicsReducedTest20.csv")

def getTrainTestSplits():
    df_train = pd.read_csv("../study/stats/datasetRadiomicsReducedTrain20.csv", index_col="ID")
    df_test = pd.read_csv("../study/stats/datasetRadiomicsReducedTest20.csv", index_col="ID")
    X_train, y_train = splitFeatureLabels(df_train)
    X_test, y_test = splitFeatureLabels(df_test)
    return X_train, X_test, y_train, y_test

def countByFeatureGroups(X_only_pyRadiomicsFeatures):
    X = X_only_pyRadiomicsFeatures
    features = pd.Series(X.columns)

    regions = features.map(lambda x: x.split("_")[0])
    pyradiomicImages = features.map(lambda x: x.split("_")[-3])
    pyradiomicFeatures = features.map(lambda x: x.split("_")[-1])

    multiIndices = pd.MultiIndex.from_arrays((regions, pyradiomicImages, pyradiomicFeatures), names=("Region", "Image", "Feature"))

    base = pd.Series(np.ones(regions.size), index=multiIndices, name="FeatureGroups")

    return base.groupby(level=["Region", "Image", "Feature"]).count()

def getVolumeDS(study_fold):
    dataset_dMRI = []
    not_patients = [5, 8, 20, 21]

    affine = []

    for i in range(1, 24): # For each subject
        if i in not_patients:
            continue

        subj = "VNSLC_%02d" % i

        subj_channels = []

        for image_model in ["_FA", "_AD", "_RD", "_MD", "icvf", "odi", "fextra", "fiso", "wFA", "wMD", "wAD", "wRD", "diamond_frac_csf", "wfvf", "mf_frac_csf"]:
            niimg = nl.image.load_img(f"{study_fold}/subjects/{subj}/dMRI/microstructure/*/*{image_model}*nii.gz")
            subj_channels.append(niimg.get_fdata())
        affine.append(niimg.affine)

        subj_channels = np.array(subj_channels)

        dataset_dMRI.append(subj_channels)

    dataset_dMRI = np.array(dataset_dMRI).squeeze()
    affine = np.array(affine)

    return dataset_dMRI, affine

#%% Transformers Extensions %%#

class MADOutlierRemotion(BaseEstimator, TransformerMixin):
    # Median Absolute Deviation
    # Private

    def __doubleMAD(self, X):
        m = X.median()
        abs_centred_median = abs(X-m)
        left = abs_centred_median[X<=m].median()
        right = abs_centred_median[X>=m].median()
        return left, right
    
    def __doubleMADsFromMedian(self, X):
        left, right = self.__doubleMAD(X)
        m = X.median()
        mad = np.copy(np.broadcast_to(left, X.shape))
        right_full = np.copy(np.broadcast_to(right, X.shape))
        mad[X>m] = right_full[X>m]
        distance = abs(X-m)/mad
        distance[X==m] = 0
        return distance

    def __init__(self, threshold_val=3) -> None:
        self.threshold_val = threshold_val
        super().__init__()

    # Public

    def fit(self, X, y=None):
        mask_outliers = self.__doubleMADsFromMedian(X) > self.threshold_val

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

class MannwhitenFilter(BaseEstimator, TransformerMixin):
    def __init__(self, pval_thresh=0.05) -> None:
        super().__init__()
        self.pval_thresh = pval_thresh
    
    def fit(self, X, y):
        X_resp = X[y==1]
        X_not = X[y==0]
        self.p_values = stats.mannwhitneyu(X_resp, X_not, nan_policy="omit").pvalue
        self.mask = self.p_values<self.pval_thresh
        return self

    def transform(self, X, y=None):
        if type(X) is pd.core.frame.DataFrame:
            return X.iloc[:, self.mask]
        else:
            return X[:, self.mask]

class KruskalFilter(BaseEstimator, TransformerMixin):
    def __init__(self, y3, pval_thresh=0.05) -> None:
        super().__init__()
        self.pval_thresh = pval_thresh
        self.y3 = y3
    
    def fit(self, X, y):
        y3 = self.y3[X.index]
        assert y3.shape == y.shape
        assert ((y3>0) == (y>0)).all()

        X_resp = X[y3==2]
        X_part = X[y3==1]
        X_not = X[y3==0]
        self.p_values = stats.kruskal(X_not, X_part, X_resp, nan_policy="omit").pvalue
        self.mask = self.p_values<self.pval_thresh
        return self

    def transform(self, X, y=None):
        if type(X) is pd.core.frame.DataFrame:
            return X.iloc[:, self.mask]
        else:
            return X[:, self.mask]

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
    
class RFECVDF(RFECV):

    def transform(self, X):
        self.columns = X.columns
        self.indices = X.index
        temp = super().transform(X)
        self.columns = self.columns[self.get_support()]
        return pd.DataFrame(temp, index=self.indices, columns=self.columns)
    
class PrintShape(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X):
        if X.shape[0] == 1:
            print(X.shape)
        return X
    
class SFSDF(SFS):
    def transform(self, X):
        self.columns = X.columns
        self.indices = X.index
        temp = super().transform(X)
        self.columns = self.columns[list(self.k_feature_idx_)]
        return pd.DataFrame(temp, index=self.indices, columns=self.columns)
    
class FilterDF(BaseEstimator, TransformerMixin):
    def __init__(self, regex=".*") -> None:
        self.regex = regex
        super().__init__()

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.filter(regex=self.regex)

#%% Estimate ML Scores %%#

def auc_and_f1(y, y_prob, **kwargs):
    isBinary = True if np.unique(y).size == 2 else False

    if isBinary:
        precision, recall, threshold = precision_recall_curve(y, y_prob, pos_label=1)

        #Find the threshold value that gives the best F1 Score
        best_f1_index = np.argmax([ calc_f1(p_r) for p_r in zip(precision, recall)])
        best_threshold, best_precision, best_recall = threshold[best_f1_index], precision[best_f1_index], recall[best_f1_index]

        # Calulcate predictions based on the threshold value
        y_pred = np.where(y_prob > best_threshold, 1, 0)
    else:
        y_pred = np.argmax(y_prob, axis=1) # Predict the one with the highest probability

    # Calculate metrics
    roc = roc_auc_score(y, y_prob, average=None if isBinary else "weighted", multi_class="raise" if isBinary else "ovo")
    f1 = f1_score(y, y_pred, pos_label=1, average="binary" if isBinary else "weighted")
    brier = 0
    if isBinary:
        brier = - brier_score_loss(y, y_prob)
    
    return roc + f1 + brier

def retScores(y_true, y_prob_decision, **kwargs):
    return y_prob_decision

def calc_f1(p_and_r):
    p, r = p_and_r
    if p == 0 and r == 0:
        return 0
    return (2*p*r)/(p+r)

def printScores(y_true, y_prob_decision, decision = False, doPrints=True, confusion=False):

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob_decision, pos_label=1)
    best_f1_index = np.argmax([ calc_f1(p_r) for p_r in zip(precisions, recalls)])
    best_threshold = thresholds[best_f1_index]

    y_pred = np.where(y_prob_decision >= best_threshold, 1, 0)

    roc = roc_auc_score(y_true, y_prob_decision)
    f1 = f1_score(y_true, y_pred)
    acc = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    accDef = balanced_accuracy_score(y_true, y_pred)
    accDef1= accuracy_score(y_true, y_pred)
    if decision is False:
        brier = brier_score_loss(y_true, y_prob_decision)
        log = log_loss(y_true, y_prob_decision)

    # Calculate and Display the confusion Matrix
    if confusion:
        cm = confusion_matrix(y_true, y_pred)

        plt.title('Confusion Matrix')
        sns.set(font_scale=1.0) #for label size
        sns.heatmap(cm, annot = True, fmt = 'd', xticklabels = ['Non responder', 'Responder'], yticklabels = ['Non responder', 'Responder'], annot_kws={"size": 14}, cmap = 'Blues')# font size

        plt.xlabel('Prediction')
        plt.ylabel('Truth')
    
    if doPrints:
        print("AUC:\t\tTest: %.3f" % (roc))
        print("f1:\t\tTest: %.3f" % (f1))
        print("Accuracy:\tTest: %.3f" % (acc))
        print("Accuracy not adjusted: %.3f %.3f" % (accDef, accDef1) )
        if decision is False:
            print("Brier:\t\tTest: %.3f" % (brier))
            print("LogLoss:\tTest %.3f" % (log))

    if decision is False:
        return {
            "auc" : roc,
            "f1" : f1,
            "accAdj": acc,
            "acc": accDef,
            "brier": brier,
            "log": log,
            "threshold": best_threshold,
            "prob_decision": list(y_prob_decision)
        }

    return {
        "auc" : roc,
        "f1" : f1,
        "accAdj": acc,
        "acc": accDef,
        "threshold": best_threshold,
        "prob_decision": list(y_prob_decision)
    }

def scoreLOO(algorithm, X, y, regex=".*", idx=None, decision=False, doPrints=True, confusion=False):

    X_filtered = X.filter(regex=regex)
    if X_filtered.shape[1] == 0:
        return None
    
    if idx is not None:
        X_filtered = X_filtered.iloc[:, list(idx)]

    pipe = Pipeline([
        # ("outliers", MADOutlierRemotion(3)),
        ("scaler", RobustScaler()),
        #("print", PrintShape()),
        ("clf", algorithm)
    ])

    cv = cross_validate(
        pipe,
        X_filtered, y,
        scoring=make_scorer(
            retScores,
            needs_proba=not decision,
            needs_threshold=decision
        ),
        cv=LeaveOneOut(),
        n_jobs=-1,
        #verbose=10,
        return_estimator=True,
        error_score=0.5
    )

    return printScores(y, cv["test_score"], decision=decision, doPrints=doPrints, confusion=confusion)

def scoreCV(algorithm, X, y, regex=".*", idx=None, decision=False, doPrints=True):

    X_filtered = X.filter(regex=regex)
    if X_filtered.shape[1] == 0:
        return None
    
    if idx is not None:
        X_filtered = X_filtered.iloc[:, list(idx)]

    pipe = Pipeline([
        # ("outliers", MADOutlierRemotion(3)),
        ("scaler", RobustScaler()),
        #("print", PrintShape()),
        ("clf", algorithm)
    ])

    cv = cross_validate(
        pipe,
        X_filtered, y,
        scoring=(
            "roc_auc",
            "f1",
            "balanced_accuracy",
            "accuracy",
            "neg_brier_score",
            "neg_log_loss",
        ) if decision is False else (
            "roc_auc",
            "f1",
            "balanced_accuracy",
            "accuracy",
        ),
        cv=StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=7,
        ),
        n_jobs=8,
        #verbose=10,
        return_estimator=True,
        error_score=0.5
    )

    roc = np.mean(cv["test_roc_auc"])
    f1 = np.mean(cv["test_f1"])
    acc = np.mean(cv["test_balanced_accuracy"])
    accNotB = np.mean(cv["test_accuracy"])
    if not decision:
        brier = np.mean(cv["test_neg_brier_score"])
        log = np.mean(cv["test_neg_log_loss"])

    if doPrints:
        print("AUC:\t\tTest: %.3f" % (roc))
        print("f1:\t\tTest: %.3f" % (f1))
        print("Accuracy not adjusted: %.3f %.3f" % (acc, accNotB) )
        if decision is False:
            print("Brier:\t\tTest: %.3f" % (brier))
            print("LogLoss:\tTest %.3f" % (log))

    if decision is False:
        return {
            "auc" : roc,
            "f1" : f1,
            "acc": acc,
            "brier": brier,
            "log": log,
        }

    return {
        "auc" : roc,
        "f1" : f1,
        "acc": acc,
    }   

# TOO UPDATE LIKE scoreLOO
def scoreLOOVoting(algorithms, X, y, regex, decision=False):

    X_filtered = X.filter(regex=regex)
    if X_filtered.shape[1] == 0:
        return None

    pipe = Pipeline([
        # ("outliers", MADOutlierRemotion(3)),
        ("scaler", RobustScaler()),
        # ("print", PrintShape()),
        ("clf", VotingClassifier(
            algorithms,
            voting="soft",
        ))
    ])

    cv = cross_validate(
        pipe,
        X_filtered, y,
        scoring=make_scorer(
            retScores,
            needs_proba=not decision,
            needs_threshold=decision
        ),
        cv=LeaveOneOut(),
        n_jobs=-1,
        verbose=10,
        return_estimator=True,
        error_score="raise"
    )

    return printScores(y, cv["test_score"])

def fitTrain_scoreTest(algorithm, DTR, DTE, LTR, LTE, regex=".*", idx=None):

    DTR_filtered = DTR.filter(regex=regex)
    DTE_filtered = DTE.filter(regex=regex)

    if DTR_filtered.shape[1] == 0:
        return None
    
    if idx is not None:
        DTR_filtered = DTR_filtered.iloc[:, list(idx)]
        DTE_filtered = DTE_filtered.iloc[:, list(idx)]
    
    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", algorithm)
    ])

    pipe.fit(DTR_filtered, LTR)
    try:
        y_prob = pipe.predict_proba(DTE_filtered)[:, 1]
        return printScores(LTE, y_prob, confusion=True)
    except:
        y_decision = pipe.decision_function(DTE_filtered)
        return printScores(LTE, y_decision, decision=True, confusion=True)
    
#%% RESULTS %%#

def printTableUniMulti(file_path, score="auc", skip_uni=None, skip_multi=None, add_uni=None):
    df = {}

    res = {}
    with open(file_path , "r") as inFile:
        res = json.load(inFile)

    prec_uni = None
    print("%20s\t" % "", end="")
    for combi in res.keys():
        if res[combi] == "Error":
            continue
        uni, multi = combi.split()
        if prec_uni != None and prec_uni != uni:
            break
        if skip_multi != None and multi in skip_multi:
            continue
        print(multi, "\t", end="")
        prec_uni = uni

    prec_uni = None
    for combi in res.keys():
        if res[combi] == "Error":
            continue
        uni, multi = combi.split()
        if skip_uni != None and uni in skip_uni:
            continue
        if prec_uni != uni:
            print()
            print("%20s\t" % uni,end="")
            df[uni] = {}
        if skip_multi != None and multi in skip_multi:
            continue
        df[uni][multi] = res[combi][score]
        print("%.4f\t" % res[combi][score], end="")
        prec_uni = uni

    if add_uni == None:
        return pd.DataFrame(df).T

    for new_uni_name, new_uni_path in add_uni.items():
        with open(new_uni_path , "r") as inFile:
            res = json.load(inFile)

        print()
        print("%20s\t" % new_uni_name,end="")
        df[new_uni_name] = {}
        for multi in res.keys():
            if res[multi] == "Error":
                continue
            if skip_multi != None and multi in skip_multi:
                continue
            print("%.4f\t" % res[multi][score], end="")
            df[new_uni_name][multi] = res[multi][score]

    return pd.DataFrame(df).T
