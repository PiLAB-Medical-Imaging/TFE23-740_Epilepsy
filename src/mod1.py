import utils
import json

from sklearn.model_selection import cross_validate, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from ITMO_FS.filters import multivariate
from sklearn.linear_model import LogisticRegressionCV
from scipy import stats

def cvMultivariateFilter(X, y, filter_name, algorithm=None, decision=False):
    if algorithm is None:
        algorithm = LogisticRegressionCV(
            cv=3,
            scoring="neg_log_loss",
            class_weight="balanced",
            random_state=7
        )

    return cross_validate(
        Pipeline([
            ("selection", Pipeline([
                ("outlier", utils.MADOutlierRemotion(3)),
                ("scaler", RobustScaler()),
                ("multivariate", multivariate.MultivariateFilter(filter_name, 15))
            ])),
            ("clf", algorithm)
        ]),
        X, y,
        scoring=make_scorer(utils.retScores, needs_proba=not decision, needs_threshold=decision),
        cv=LeaveOneOut(),
        n_jobs=4,
        verbose=2,
        error_score="raise",
        # return_estimator=True Da problemi nel salvare il json
    )

def runMod1(X, y):

    cv = cvMultivariateFilter(X, y, "MRMR")

    return utils.printScores(y, cv["test_score"], confusion=True)

def runMod2(X, y):
    cvs = {}
    for filter_name in ["CIFE","CFR","MRI","IWFS"]:
        try:
            print(filter_name)
            cv = cvMultivariateFilter(X, y, filter_name)
            cvs[filter_name] = utils.printScores(y, cv["test_score"])
        except:
            print("Error: ", filter_name)
    return cvs

def runMod3(X, y):
    for name, algorithm in [
        ("logreg", LogisticRegressionCV(random_state=7, class_weight="balanced", scoring="roc_auc", cv=3, n_jobs=1)),
        ("svm", RandomizedSearchCV(
            SVC(C=1e-6, random_state=7, class_weight="balanced"),
            param_distributions={
                "C": stats.loguniform(1e-6, 1)
            },
            scoring="roc_auc", cv=3, n_jobs=1, random_state=7
        )),
        ("knn", GridSearchCV(
            CalibratedClassifierCV(KNeighborsClassifier(), cv=3),
            param_grid={
                "n_neighbors": [2, 3, 4, 5, 6]
            },
            scoring="roc_auc", cv=3, n_jobs=1
        )),
        ("mlp", RandomizedSearchCV(
            MLPClassifier(random_state=7, learning_rate="adaptive", max_iter=1000),
            param_distributions={
                "alpha": stats.loguniform(1e-3, 1e1),
                "hidden_layer_sizes": [(100,), (50,), (20,)]
            },
            scoring="roc_auc", cv=3, n_jobs=1, random_state=7
        )),
        ("gauss", GaussianNB())
    ]:
        cvs = {}
        for filter_name in ["CIFE","CFR","MRI"]: # IWFS ???
            try:
                print(filter_name)
                decision = True if name == "svm" else False
                cv = cvMultivariateFilter(X, y, filter_name, algorithm=algorithm, decision=decision)
                cvs[filter_name] = utils.printScores(y, cv["test_score"], decision=decision)
            except:
                print("Error: ", filter_name)
                cvs[filter_name] = {filter_name: "Error"}
        
        with open(f"../study/stats/results-{name}-loo-filter.json", "w") as outfile:
            json.dump(cvs, outfile, indent=2, sort_keys=True)
            
def main():
    
    df = utils.getReducedDS()
    X, y, y3 = utils.splitFeatureLabels(df)

    runMod3(X, y)

if __name__ == "__main__":
    exit(main())
