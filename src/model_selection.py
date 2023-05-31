import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from params import *

from myTransformers import FilterSmall
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression, ElasticNet, LinearRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score

sns.set_theme()

def myScoreFunc(X, y):
    mi_score = mutual_info_classif(X, y)
    f_score = f_classif(X, y)[0]
    return mi_score + f_score

def model_selection(folder_path):

    stats_path = folder_path + "/stats"

    # Reading the whole dataset
    df = pd.read_csv("%s/dataset.csv" % stats_path, index_col="ID")
    print("Before:", df.shape)

    # Removing the NaN values
    df = df.dropna(axis=0, how="any")

    # Separing Data from labels and removing benzo and type
    X = df.drop(["resp", "respPart"], axis=1)
    y = df["resp"]

    # splitting training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, shuffle=True, stratify=y)
    print("X_train", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train", y_train.shape)
    print("y_test", y_test.shape)
    print(y_test)

    # Removing to small dMRI features
    filtr = FilterSmall(1e-12, X_train.apply(lambda x : type(x[0]) == np.int64, axis=0))
    filtr.fit(X_train) 
    X_train = pd.DataFrame(filtr.transform(X_train), index=X_train.index, columns=X_train.columns[filtr.get_support()])
    X_test = pd.DataFrame(filtr.transform(X_test), index=X_test.index, columns=X_test.columns[filtr.get_support()])

    col_dMRI = X_train.filter(regex=r'mean|std|skew|kurt').columns
    col_nTract = X_train.filter(regex=r'nTracts').columns
    col_cont = ["age", "therapy_duration", "epilepsy_onset_age", "epilepsy_duration", *col_nTract, *col_dMRI]
    col_disc = ["sex", "AEDs", "benzo", "epilepsy_type"]
    print("X_train After:", X_train.shape)

    models = {
        # SVM
        "linearSVM" : (
            LinearSVC(), 
            {
                "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
                "classifier__C" : (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1),
            }
        ),
        "SVM" : (
            SVC(),
            [
                {
                    "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
                    "classifier__C" : (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1),
                    "classifier__kernel" : ['linear'], 
                },
                {
                    "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
                    "classifier__C" : (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1),
                    "classifier__kernel" : ['poly'],
                    "classifier__degree" : (2, 3, 4),
                },
                {
                    "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
                    "classifier__C" : (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1),
                    "classifier__kernel" : ['rbf'], 
                    "classifier__gamma" : (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1),
                },
            ]
        ),
        # Linear 
        "LogReg" : (
            LogisticRegression(),
            {
                "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
                "classifier__penalty" : ["l2"],
                "classifier__dual" : [True],
                "classifier__C" :(1e-3, 1e-2, 1e-1),
            },
        ),
        "LinReg" : (
            LinearRegression(),
            {
                "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
            }
        ),
        "ridgeReg" : (
            RidgeClassifier(),
            {
                "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
                "classifier__alpha" : (0.5, 1, 5, 10, 20, 40)
            }
        ),
        "elasticNet" : (
            ElasticNet(),
            {
                "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
                "classifier__alpha" : (0.5, 1, 5, 10, 20, 40),  # 0 == Linear Regression
                "classifier__l1_ratio" : (0.5, 1), # 0 == Ridge Regression, 1 == Lasso Regression
            },
        ),
        # Nearest Neighbors
        "neighbors" : (
            KNeighborsClassifier(),
            {
                "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
                "classifier__n_neighbors" : (2, 3, 5),
                "classifier__weights" : ("uniform", "distance"),
            }
        ),
        # Naive Bayes
        "gaussianNaive" : (
            GaussianNB(),
            {
                "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
            }
        ),
        "multinomialNaive" : (
            MultinomialNB(),
            {
                "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
            }
        ),
        "complementNaive" : (
            ComplementNB(),
            {
                "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
            }
        ),
        "bernulliNaive" : (
            BernoulliNB(),
            {
                "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
            }
        ),
        "categoricalNaive" : (
            CategoricalNB(),
            {
                "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
            }
        ),
        # Tree
        "tree" : (
            DecisionTreeClassifier(),
            {
                "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
                "classifier__criterion" : ("gini", "entropy", "log_loss")
                # ccp_apha is a Regularization therm (to try)
            }
        ),
        # Ensemble
        "forest" : (
            RandomForestClassifier(),
            {
                "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
                "classifier__n_estimators" : (100, 500, 1000),
                # Check if make sense choose also the criterion (see the tree)
                "classifier__bootstrap" : [True],
                "classifier__max_samples" : [0.5],
                "classifier__max_features" : ["log2"],
                "classifier__warm_star" : [True],
                "classifier__oob_score" : [True],
                "classifier__max_depth" : [10, 20, 30],
                # ccp_alpha da checkare
            }
        ),
        "extraForest" : (
            ExtraTreesClassifier(),
            {
                "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
                "classifier__n_estimators" : (100, 500, 1000),
                # Check if make sense choose also the criterion (see the tree)
                "classifier__bootstrap" : [True],
                "classifier__max_samples" : [0.5],
                "classifier__max_features" : ["log2"],
                "classifier__warm_star" : [True],
                "classifier__oob_score" : [True],
                "classifier__max_depth" : [10, 20, 30],
                # ccp_alpha da checkare
            }
        ),
        "gradientBoosting": (
            GradientBoostingClassifier(),
            {
                "selection__k" : (2, 3, 5, 7, 9, 11, 13, 17, 19),
                "classifier__n_estimators" : (100, 200, 500, 1000),
                "classifier__learning_rate" : (0.05, 0.1),
                # Check if make sense choose also the criterion (see the tree)
                "classifier__max_samples" : [0.5],
                "classifier__max_features" : ["log2"],
                "classifier__warm_star" : [True],
                "classifier__max_depth" : [10, 20, 30],
                "classifier__validation_fraction" : [0.20],
                "classifier__n_iter_no_change" : [50],
                # ccp_alpha da checkare
            }
        )
    }

    sss = StratifiedShuffleSplit(n_splits=20, test_size=1/3)

    fitted_models = {}

    for modelName, (classifier, grid) in models.items():
        for scaler in [StandardScaler(), RobustScaler()]:
            for filter in [f_classif, mutual_info_classif, myScoreFunc]:
                print("modelName:", modelName, ", scaler:", scaler.__class__.__name__, ", filter:", filter.__name__)

                # Scaling
                pre = ColumnTransformer(
                        [("scaling", scaler, col_cont)],
                        remainder="passthrough", # one hot or other stuff
                        n_jobs=-1
                    )

                # Remove costant values
                # Yes, it's possible, since we have few data after the splitting and the cross validation is possible to have some features with same values. To remove them from the modeling we use the 
                varThres = VarianceThreshold()

                # Feature Selection
                selection = SelectKBest(
                        score_func=filter,
                    )

                # Pipe
                pipe = Pipeline([
                    ("pre", pre),
                    ("varThres", varThres),
                    ("selection", selection),
                    ("classifier", classifier)
                ])

                model = GridSearchCV(
                    estimator=pipe,
                    param_grid=grid,
                    scoring="roc_auc",
                    n_jobs=-1,
                    cv=sss,
                )

                model.fit(X_train, y_train)
                fitted_models[modelName] = model

                # Train
                print(modelName, "train score:", model.best_score_)
                print(model.best_params_) 

                # Test
                y_pred = model.predict(X_test)
                confusionMatrix = confusion_matrix(y_test, y_pred)  
                print(confusionMatrix)  
                print("Balanced Accuracy Score", balanced_accuracy_score(y_test, y_pred, adjusted=True))
                print("Area Under ROC", roc_auc_score(y_test, y_pred,))

def main():
    ## Getting folder
    folder_path = get_folder(sys.argv)

    model_selection(folder_path)

if __name__ == "__main__":
    exit(main())
