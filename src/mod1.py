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
from ITMO_FS.filters import multivariate, univariate
from sklearn.linear_model import LogisticRegressionCV
from scipy import stats
from feature_engine.selection import SmartCorrelatedSelection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

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

listOfAlgorithms = [
    ("logreg", LogisticRegressionCV(random_state=7, class_weight="balanced", scoring="neg_log_loss", cv=3, n_jobs=1)),
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
            "estimator__n_neighbors": [2, 3, 4, 5, 6]
        },
        scoring="neg_log_loss", cv=3, n_jobs=1
    )),
    ("mlp", RandomizedSearchCV(
        MLPClassifier(random_state=7, learning_rate="adaptive", max_iter=1000),
        param_distributions={
            "alpha": stats.loguniform(1e-3, 1e1),
            "hidden_layer_sizes": [(100,), (50,), (20,)]
        },
        scoring="neg_log_loss", cv=3, n_jobs=1, random_state=7
    )),
    ("gauss", GaussianNB())
]

def runMod3(X, y):
    for name, algorithm in listOfAlgorithms:
        print(name)
        cvs = {}
        for filter_name in ["CIFE","CFR","MRI"]:
            try:
                print(filter_name)
                decision = name == "svm"
                cv = cvMultivariateFilter(X, y, filter_name, algorithm=algorithm, decision=decision)
                cvs[filter_name] = utils.printScores(y, cv["test_score"], decision=decision)
            except:
                print("Error: ", filter_name)
                cvs[filter_name] = {filter_name: "Error"}
        
        with open(f"../study/stats/results-{name}-loo-filter.json", "w") as outfile:
            json.dump(cvs, outfile, indent=2, sort_keys=True)

def runMod4(X, y, y3):
    print("Mod4")

    for name, algorithm in listOfAlgorithms:
        print(name)
        decision = name == "svm"
        cv = cross_validate(
            Pipeline([
                ("selection", Pipeline([
                    ("outlier", utils.MADOutlierRemotion(3)),
                    ("scaler", RobustScaler()),
                    ("t-test", utils.MannwhitenFilter(0.05)),
                    ("ANOVA", utils.KruskalFilter(y3, 0.05)),
                    ("correlated", SmartCorrelatedSelection(threshold=0.95,missing_values="raise", selection_method="variance")),
                    ("print", utils.PrintShape())
                ])),
                ("clf", algorithm)
            ]),
            X, y,
            scoring=make_scorer(utils.retScores, needs_proba=not decision, needs_threshold=decision),
            cv=LeaveOneOut(),
            n_jobs=4,
            verbose=2,
            error_score="raise",
        )

        res = utils.printScores(y, cv["test_score"], decision=decision)
        with open(f"../study/stats/results-{name}-loo-filter-ManKru.json", "w") as outfile:
            json.dump(res, outfile, indent=2, sort_keys=True)

def runMod5(X, y ,y3):
    print("Mod5")

    for name, algorithm in listOfAlgorithms:
        print(name)
        decision = name == "svm"
        cvs = {}
        for filter_name in ["MIM","MRMR","JMI","CIFE","MIFS","CMIM","ICAP","DCSF","CFR","MRI","IWFS"]:
            try:
                print(filter_name)
                cv = cross_validate(
                    Pipeline([
                        ("selection", Pipeline([
                            ("outlier", utils.MADOutlierRemotion(3)),
                            ("scaler", RobustScaler()),
                            ("t-test", utils.MannwhitenFilter(0.05)),
                            ("ANOVA", utils.KruskalFilter(y3, 0.05)),
                            ("correlated", SmartCorrelatedSelection(threshold=0.95,missing_values="raise", selection_method="variance")),
                            ("multivariate", multivariate.MultivariateFilter(filter_name, 15)),
                        ])),
                        ("clf", algorithm)
                    ]),
                    X, y,
                    scoring=make_scorer(utils.retScores, needs_proba=not decision, needs_threshold=decision),
                    cv=LeaveOneOut(),
                    n_jobs=4,
                    verbose=2,
                    error_score="raise",
                )
                cvs[filter_name] = utils.printScores(y, cv["test_score"], decision=decision)
            except:
                print("Error: ", filter_name)
                cvs[filter_name] = "Error"

        with open(f"../study/stats/results-{name}-loo-filter-ManKru-Multi.json", "w") as outfile:
                json.dump(cvs, outfile, indent=2, sort_keys=True)

def runMod5_1(X, y):
    print("Mod5.1")

    for name, algorithm in listOfAlgorithms:
        print(name)
        decision = name == "svm"
        cvs = {}
        for filter_name in ["MIM","MRMR","JMI","CIFE","MIFS","CMIM","ICAP","DCSF","CFR","MRI","IWFS"]:
            try:
                print(filter_name)
                cv = cross_validate(
                    Pipeline([
                        ("selection", Pipeline([
                            ("outlier", utils.MADOutlierRemotion(3)),
                            ("scaler", RobustScaler()),
                            ("t-test", utils.MannwhitenFilter(0.05)),
                            ("correlated", SmartCorrelatedSelection(threshold=0.95,missing_values="raise", selection_method="variance")),
                            ("multivariate", multivariate.MultivariateFilter(filter_name, 15)),
                        ])),
                        ("clf", algorithm)
                    ]),
                    X, y,
                    scoring=make_scorer(utils.retScores, needs_proba=not decision, needs_threshold=decision),
                    cv=LeaveOneOut(),
                    n_jobs=4,
                    verbose=2,
                    error_score="raise",
                )
                cvs[filter_name] = utils.printScores(y, cv["test_score"], decision=decision)
            except:
                print("Error: ", filter_name)
                cvs[filter_name] = "Error"

        with open(f"../study/stats/results-{name}-loo-filter-Man-Multi.json", "w") as outfile:
                json.dump(cvs, outfile, indent=2, sort_keys=True)

def runMod6(X, y):
    print("Mod6")

    for name, algorithm in listOfAlgorithms:
        print(name)
        decision = name == "svm"
        cvs = {}
        for filter_uni in [
            univariate.f_ratio_measure,
            univariate.gini_index,
            univariate.su_measure,
            univariate.spearman_corr,
            univariate.pearson_corr,
            univariate.fechner_corr,
            univariate.kendall_corr,
            univariate.reliefF_measure,
            univariate.chi2_measure,
            univariate.information_gain
        ]:
            
            for filter_multi_name in ["MIM","MRMR","JMI","CIFE","MIFS","CMIM","ICAP","DCSF","CFR","MRI","IWFS"]:
                try:
                    union_names = filter_uni.__name__ + " " + filter_multi_name
                    print(union_names)
                    cv = cross_validate(
                        Pipeline([
                            ("selection", Pipeline([
                                ("outlier", utils.MADOutlierRemotion(3)),
                                ("scaler", RobustScaler()),
                                ("univariate", univariate.UnivariateFilter(filter_uni, univariate.select_k_best(1000))),
                                ("multivariate", multivariate.MultivariateFilter(filter_multi_name, 15)),
                            ])),
                            ("clf", algorithm)
                        ]),
                        X, y,
                        scoring=make_scorer(utils.retScores, needs_proba=not decision, needs_threshold=decision),
                        cv=LeaveOneOut(),
                        n_jobs=-1,
                        verbose=2,
                        error_score="raise",
                    )
                    cvs[union_names] = utils.printScores(y, cv["test_score"], decision=decision)
                except:
                    print("Error: ", union_names)
                    cvs[union_names] = "Error"

        with open(f"../study/stats/results-{name}-loo-filter-Uni-Multi.json", "w") as outfile:
                json.dump(cvs, outfile, indent=2, sort_keys=True)

def runMod7(X, y):
    print("Mod7")

    for name, algorithm in listOfAlgorithms:
        print(name)
        decision = name == "svm"
        cvs = {}
        for filter_uni in [
            univariate.f_ratio_measure,
            univariate.spearman_corr,
            univariate.kendall_corr,
            univariate.reliefF_measure,
        ]:
            
            for filter_multi_name in ["MRMR","CIFE","ICAP","DCSF","CFR","MRI"]:
                try:
                    union_names = filter_uni.__name__ + " " + filter_multi_name
                    print(union_names)
                    cv = cross_validate(
                        Pipeline([
                            ("selection", Pipeline([
                                ("outlier", utils.MADOutlierRemotion(3)),
                                ("scaler", RobustScaler()),
                                ("univariate", univariate.UnivariateFilter(filter_uni, univariate.select_k_best(1000))),
                                ("multivariate", multivariate.MultivariateFilter(filter_multi_name, 20)),
                                ("sfs", SFS(
                                    algorithm,
                                    k_features=(1,15),
                                    floating=True,
                                    scoring="roc_auc",
                                    cv=3,
                                ))
                            ])),
                            ("clf", algorithm)
                        ]),
                        X, y,
                        scoring=make_scorer(utils.retScores, needs_proba=not decision, needs_threshold=decision),
                        cv=LeaveOneOut(),
                        n_jobs=-1,
                        verbose=2,
                        error_score="raise",
                    )
                    cvs[union_names] = utils.printScores(y, cv["test_score"], decision=decision)
                except:
                    print("Error: ", union_names)
                    cvs[union_names] = "Error"

        with open(f"../study/stats/results-{name}-loo-filter-Uni-Multi-sfs.json", "w") as outfile:
                json.dump(cvs, outfile, indent=2, sort_keys=True)

def runMod7_1(X, y):
    print("Mod7_1")

    for name, algorithm in listOfAlgorithms:
        print(name)
        decision = name == "svm"
        cvs = {}
        for filter_name in ["MRMR","CIFE","ICAP","DCSF","CFR","MRI"]:
            try:
                print(filter_name)
                cv = cross_validate(
                    Pipeline([
                        ("selection", Pipeline([
                            ("outlier", utils.MADOutlierRemotion(3)),
                            ("scaler", RobustScaler()),
                            ("t-test", utils.MannwhitenFilter(0.05)),
                            ("correlated", SmartCorrelatedSelection(threshold=0.95,missing_values="raise", selection_method="variance")),
                            ("multivariate", multivariate.MultivariateFilter(filter_name, 20)),
                            ("sfs", SFS(
                                algorithm,
                                k_features=(1,15),
                                floating=True,
                                scoring="roc_auc",
                                cv=3,
                            ))
                        ])),
                        ("clf", algorithm)
                    ]),
                    X, y,
                    scoring=make_scorer(utils.retScores, needs_proba=not decision, needs_threshold=decision),
                    cv=LeaveOneOut(),
                    n_jobs=-1,
                    verbose=2,
                    error_score="raise",
                )
                cvs[filter_name] = utils.printScores(y, cv["test_score"], decision=decision)
            except:
                print("Error: ", filter_name)
                cvs[filter_name] = "Error"

        with open(f"../study/stats/results-{name}-loo-filter-Man-Multi-sfs.json", "w") as outfile:
                json.dump(cvs, outfile, indent=2, sort_keys=True)

def runMod7_2(X, y, y3):
    print("Mod7_2")

    for name, algorithm in listOfAlgorithms:
        print(name)
        decision = name == "svm"
        cvs = {}
        for filter_name in ["MRMR","CIFE","ICAP","DCSF","CFR","MRI"]:
            try:
                print(filter_name)
                cv = cross_validate(
                    Pipeline([
                        ("selection", Pipeline([
                            ("outlier", utils.MADOutlierRemotion(3)),
                            ("scaler", RobustScaler()),
                            ("t-test", utils.MannwhitenFilter(0.05)),
                            ("ANOVA", utils.KruskalFilter(y3, 0.05))
                            ("correlated", SmartCorrelatedSelection(threshold=0.95,missing_values="raise", selection_method="variance")),
                            ("multivariate", multivariate.MultivariateFilter(filter_name, 20)),
                            ("sfs", SFS(
                                algorithm,
                                k_features=(1,15),
                                floating=True,
                                scoring="roc_auc",
                                cv=3,
                            ))
                        ])),
                        ("clf", algorithm)
                    ]),
                    X, y,
                    scoring=make_scorer(utils.retScores, needs_proba=not decision, needs_threshold=decision),
                    cv=LeaveOneOut(),
                    n_jobs=-1,
                    verbose=2,
                    error_score="raise",
                )
                cvs[filter_name] = utils.printScores(y, cv["test_score"], decision=decision)
            except:
                print("Error: ", filter_name)
                cvs[filter_name] = "Error"

        with open(f"../study/stats/results-{name}-loo-filter-Man-Multi-sfs.json", "w") as outfile:
                json.dump(cvs, outfile, indent=2, sort_keys=True)
            
def main():
    
    df = utils.getReducedDS()
    X, y, y3 = utils.splitFeatureLabels(df)

    runMod7(X, y)

if __name__ == "__main__":
    exit(main())
