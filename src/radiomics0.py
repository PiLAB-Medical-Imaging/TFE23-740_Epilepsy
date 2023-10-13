import utils
import json
import pandas as pd

from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures

from sklearn.model_selection import cross_validate, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from ITMO_FS.filters import multivariate, univariate
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
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
    #("logreg", LogisticRegressionCV(random_state=7, class_weight="balanced", scoring="neg_log_loss", cv=3, n_jobs=1)),
    ("svm", RandomizedSearchCV(
        SVC(random_state=7, class_weight="balanced"),
        param_distributions={
            "C": stats.loguniform(1e-6, 1e-2)
        },
        scoring="roc_auc", cv=3, n_jobs=1, random_state=7
    )),
    # ("knn", GridSearchCV(
    #     KNeighborsClassifier(),
    #     param_grid={
    #         "n_neighbors": [2, 3, 4]
    #     },
    #     scoring="neg_log_loss", cv=3, n_jobs=1
    # )),
    # ("mlp", RandomizedSearchCV(
    #     MLPClassifier(random_state=7, learning_rate="adaptive", max_iter=1000),
    #     param_distributions={
    #         "alpha": stats.loguniform(1e-3, 1e1),
    #         "hidden_layer_sizes": [(100,), (50,), (20,)]
    #     },
    #     scoring="neg_log_loss", cv=3, n_jobs=1, random_state=7
    # )),
    # ("gauss", GaussianNB())
]

listOfAlgorithmsNoCV = {
    "logreg": LogisticRegression(random_state=7, class_weight="balanced", n_jobs=1),
    "svm": SVC(C=1e-6, random_state=7, class_weight="balanced"),
    "knn": KNeighborsClassifier(),
    "mlp": MLPClassifier(random_state=7, learning_rate="adaptive", max_iter=1000),
    "gauss": GaussianNB()
}

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
                                    listOfAlgorithmsNoCV[name],
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
                        n_jobs=9,
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
                                listOfAlgorithmsNoCV[name],
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
                    n_jobs=4,
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
                            ("ANOVA", utils.KruskalFilter(y3, 0.05)),
                            ("correlated", SmartCorrelatedSelection(threshold=0.95,missing_values="raise", selection_method="variance")),
                            ("multivariate", multivariate.MultivariateFilter(filter_name, 20)),
                            ("sfs", SFS(
                                listOfAlgorithmsNoCV[name],
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
                    n_jobs=4,
                    verbose=2,
                    error_score="raise",
                )
                cvs[filter_name] = utils.printScores(y, cv["test_score"], decision=decision)
            except:
                print("Error: ", filter_name)
                cvs[filter_name] = "Error"

        with open(f"../study/stats/results-{name}-loo-filter-ManKru-Multi-sfs.json", "w") as outfile:
                json.dump(cvs, outfile, indent=2, sort_keys=True)

listOfAlgorithmsOld = [
    ("logreg", LogisticRegressionCV(random_state=7, class_weight="balanced", scoring="neg_log_loss", cv=3, n_jobs=1)),
    ("linear-svm", RandomizedSearchCV(
        LinearSVC(dual=True, random_state=7, class_weight="balanced"),
        param_distributions={
            "C": stats.loguniform(1e-6, 1)
        },
        scoring="roc_auc", cv=3, n_jobs=1, random_state=7
    )),
    ("rbf-svm", RandomizedSearchCV(
        SVC(random_state=7, class_weight="balanced"),
        param_distributions={
            "C": stats.loguniform(1e-6, 1),
            "gamma" : stats.loguniform(1e-6, 1)
        },
        scoring="roc_auc", cv=3, n_jobs=1, random_state=7
    )),
    ("poly-svm", RandomizedSearchCV(
        SVC(kernel="poly", random_state=7, class_weight="balanced"),
        param_distributions={
            "C": stats.loguniform(1e-6, 1),
            "degree": [2, 3, 4]
        },
        scoring="roc_auc", cv=3, n_jobs=1, random_state=7
    )),
]

listOfAlgorithmsNoCVOld = {
    "logreg": LogisticRegression(random_state=7, class_weight="balanced", n_jobs=1),
    "linear-svm": LinearSVC(dual=True, C=1e-6, random_state=7, class_weight="balanced"),
    "rbf-svm": SVC(C=1e-6, random_state=7, class_weight="balanced"),
    "poly-svm": SVC(C=1e-6, kernel="poly", random_state=7, class_weight="balanced")
}

def runMod8(X, y):
    print("mod8")

    for name, algorithm in listOfAlgorithmsOld:
        print(name)
        decision = "svm" in name
        cvs = {}
        #try:
        cv = cross_validate(
            Pipeline([
                ("selection", Pipeline([
                    ("outlier", utils.MADOutlierRemotion(3)),
                    ("scaler", StandardScaler()),
                    ("sfs", SFS(
                        listOfAlgorithmsNoCVOld[name],
                        k_features=(1,3),
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
            n_jobs=8,
            verbose=2,
            error_score="raise",
        )
        cvs[name] = utils.printScores(y, cv["test_score"], decision=decision)
        # except:
        #     print("Error: ", name)
        #     cvs[name] = "Error"

        with open(f"../study/stats/results-{name}-loo-first-mean-sfs.json", "w") as outfile:
                json.dump(cvs, outfile, indent=2, sort_keys=True)


def getFirstDS():
    df = pd.read_csv("../study/stats/dataset_thres1_1corr.csv", index_col="ID")
    df = df.drop(["VNSLC_16"])
    df = df.dropna(axis=1)
    df = df.drop(df.filter(regex=r'(min|max)'), axis=1) # remove max and min features because are full of imperfections
    df = df.drop(df.filter(regex=r'(_c0_|_c1_|_f0_|_f1_|_csf_mf_|_csf_d_|_fiso_|nTracts|voxVol)'), axis=1)

    pipe = Pipeline([
        ("constantFeatures", DropConstantFeatures(tol=0.62)),
        ("duplicateFeatures", DropDuplicateFeatures()),
    ])
    df = pipe.fit_transform(df)
    return df

def main():
    
    df = utils.getReducedDS()
    # df = getFirstDS()
    # 
    X, y, y3 = utils.splitFeatureLabels(df)
    # X = X.filter(regex=r'mean')

    runMod7(X, y)

if __name__ == "__main__":
    exit(main())
