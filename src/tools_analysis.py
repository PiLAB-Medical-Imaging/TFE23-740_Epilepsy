import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, balanced_accuracy_score, brier_score_loss, log_loss
import seaborn as sns
import numpy as np
sns.set_palette("muted")
    

def calc_f1(p_and_r):
    p, r = p_and_r
    if p == 0 and r == 0:
        return 0
    return (2*p*r)/(p+r)

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
    # logLoss = - log_loss(y, y_prob)
    
    return roc + f1 + brier # + logLoss


# Print the F1, Precision, Recall, ROC-AUC, and Accuracy Metrics 
# Since we are optimizing for F1 score - we will first calculate precision and recall and 
# then find the probability threshold value that gives us the best F1 score

def print_model_metrics(y_test, y_test_prob, isBinary, confusion = False, verbose = True, return_metrics = False):
    # Calculate all metrics
    if isBinary:
        precision, recall, threshold = precision_recall_curve(y_test, y_test_prob, pos_label = 1)

        # Find the threshold value that gives the best F1 Score
        best_f1_index =np.argmax([calc_f1(p_r) for p_r in zip(precision, recall)])
        best_threshold, best_precision, best_recall = threshold[best_f1_index], precision[best_f1_index], recall[best_f1_index]

        # Calulcate predictions based on the threshold value
        y_test_pred = np.where(y_test_prob > best_threshold, 1, 0)
    else:
        y_test_pred = np.argmax(y_test_prob, axis=1) # Predict the one with the highest probability

        best_precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=0.0)
        best_recall = recall_score(y_test, y_test_pred, average="weighted")

    f1 = f1_score(y_test, y_test_pred, pos_label = 1, average = "binary" if isBinary else "weighted")
    roc_auc = roc_auc_score(y_test, y_test_prob, average=None if isBinary else "weighted", multi_class="raise" if isBinary else "ovo")
    acc = accuracy_score(y_test, y_test_pred)
    b_acc = balanced_accuracy_score(y_test, y_test_pred)
    
    
    if confusion:
        # Calculate and Display the confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)

        plt.title('Confusion Matrix')
        sns.set(font_scale=1.0) #for label size
        sns.heatmap(cm, annot = True, fmt = 'd', xticklabels = ['No Clickbait', 'Clickbait'], yticklabels = ['No Clickbait', 'Clickbait'], annot_kws={"size": 14}, cmap = 'Blues')# font size

        plt.xlabel('Truth')
        plt.ylabel('Prediction')
        
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} | B_Acc: {:.3f} \n'.format(f1, best_precision, best_recall, roc_auc, acc, b_acc))
    
    if return_metrics:
        return np.array([f1, best_precision, best_recall, roc_auc, acc, b_acc])
    
    

# Run Simple Log Reg Model and Print metrics
from sklearn.linear_model import LogisticRegression

from pandas import DataFrame

def run_log_reg_cv(features: DataFrame, y: DataFrame, cv, pipeline=None, C = 1, penalty="l2", confusion = False, return_f1 = False, verbose = True):
    metrics = np.zeros(6)
    features = features.to_numpy()
    y = y.to_numpy()
    isBinary = True if np.unique(y).size == 2 else False
    for train, test in cv.split(features, y):
        train_features, test_features = features[train, :], features[test, :]
        y_train, y_test = y[train], y[test]

        if pipeline is not None:
            pipeline.fit(train_features)
            train_features = pipeline.transform(train_features)
            test_features = pipeline.transform(test_features)

        if isBinary:
            dual = True
            if penalty == "l1":
                dual=False
            # Binary case
            log_reg = LogisticRegression(
                penalty=penalty,
                dual=dual,
                C = C,
                class_weight="balanced",
                random_state=7,
                solver="liblinear",
                max_iter=100000,
            )
        else:
            # Multiclass case
            log_reg = LogisticRegression(
                penalty=penalty,
                C = C,
                class_weight="balanced",
                random_state=7,
                solver="lbfgs",
                multi_class="multinomial",
                max_iter=100000,
            )

        log_reg.fit(train_features, y_train)
        
        y_test_prob = log_reg.predict_proba(test_features)[:,1] if isBinary else log_reg.predict_proba(test_features)
        metrics += print_model_metrics(y_test, y_test_prob, isBinary, confusion = confusion, verbose = False, return_metrics = True)

    metrics /=cv.get_n_splits()
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} | B_Acc: {:.3f} \n'.format(*metrics))
    if return_f1:
        return metrics[0]
    return log_reg

from sklearn.svm import LinearSVC
def run_svm_cv(features: DataFrame, y: DataFrame, cv, pipeline=None, C = 1, penalty="l2", confusion = False, return_f1 = False, verbose = True):
    metrics = np.zeros(6)
    features = features.to_numpy()
    y = y.to_numpy()
    for train, test in cv.split(features, y):
        train_features, test_features = features[train, :], features[test, :]
        y_train, y_test = y[train], y[test]

        if pipeline is not None:
            pipeline.fit(train_features)
            train_features = pipeline.transform(train_features)
            test_features = pipeline.transform(test_features)

        dual = True
        if penalty == "l1":
            dual=False
        svm = LinearSVC(
            penalty=penalty,
            dual=dual,
            C = C,
            class_weight="balanced",
            random_state=7,
            max_iter=100000,
        )

        svm.fit(train_features, y_train)

        y_test_scores = svm.decision_function(test_features)
        metrics += print_model_metrics(y_test, y_test_scores, True, confusion = confusion, verbose = False, return_metrics = True)

    metrics /=cv.get_n_splits()
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} | B_Acc: {:.3f}  \n'.format(*metrics))
    if return_f1:
        return metrics[0]
    return svm

from sklearn.svm import SVC
def run_svm_kernel_cv(features: DataFrame, y: DataFrame, cv, pipeline=None, C = 1, kernel = "rbf", degree = 2, gamma="scale", confusion = False, return_f1 = False, verbose = True):
    metrics = np.zeros(6)
    features = features.to_numpy()
    y = y.to_numpy()
    isBinary = True if np.unique(y).size == 2 else False
    for train, test in cv.split(features, y):
        train_features, test_features = features[train, :], features[test, :]
        y_train, y_test = y[train], y[test]

        if pipeline is not None:
            pipeline.fit(train_features)
            train_features = pipeline.transform(train_features)
            test_features = pipeline.transform(test_features)

        if isBinary:
            svm = SVC(
                C = C,
                kernel=kernel,
                degree = degree,
                gamma = gamma,
                class_weight="balanced",
                max_iter=-1,
                random_state=7
            )
        else:
            svm = SVC(
                probability=True,
                C = C,
                kernel=kernel,
                degree = degree,
                gamma = gamma,
                class_weight="balanced",
                max_iter=-1,
                decision_function_shape="ovo",
                random_state=7
            )

        svm.fit(train_features, y_train)

        y_test_scores = svm.decision_function(test_features) if isBinary else svm.predict_proba(test_features)
        metrics += print_model_metrics(y_test, y_test_scores, isBinary, confusion = confusion, verbose = False, return_metrics = True)

    metrics /=cv.get_n_splits()
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} | B_Acc: {:.3f}  \n'.format(*metrics))
    if return_f1:
        return metrics[0]
    return svm

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
def run_gaussian_cv(features: DataFrame, y: DataFrame, cv, pipeline=None, confusion = False, return_f1 = False, verbose = True):
    metrics = np.zeros(6)
    features = features.to_numpy()
    y = y.to_numpy()
    isBinary = True if np.unique(y).size == 2 else False
    for train, test in cv.split(features, y):
        train_features, test_features = features[train, :], features[test, :]
        y_train, y_test = y[train], y[test]

        if pipeline is not None:
            pipeline.fit(train_features)
            train_features = pipeline.transform(train_features)
            test_features = pipeline.transform(test_features)

        gaussian = GaussianNB()
        gaussian.fit(train_features, y_train)

        y_test_prob = gaussian.predict_proba(test_features)[:, 1] if isBinary else gaussian.predict_proba(test_features)
        metrics += print_model_metrics(y_test, y_test_prob, isBinary, confusion = confusion, verbose = False, return_metrics = True)

    metrics /=cv.get_n_splits()
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} | B_Acc: {:.3f}  \n'.format(*metrics))
    if return_f1:
        return metrics[0]
    return gaussian

from sklearn.mixture import GaussianMixture
def run_mixture_cv(features: DataFrame, y: DataFrame, cv, pipeline=None, nComponents=1, covarianceType = "full", confusion = False, return_f1 = False, verbose = True):
    metrics = np.zeros(5)
    features = features.to_numpy()
    y = y.to_numpy()
    for train, test in cv.split(features, y):
        train_features, test_features = features[train, :], features[test, :]
        y_train, y_test = y[train], y[test]

        if pipeline is not None:
            pipeline.fit(train_features)
            train_features = pipeline.transform(train_features)
            test_features = pipeline.transform(test_features)

        mixture = GaussianMixture(
            n_components = nComponents,
            covariance_type = covarianceType,
            n_init=5,
            init_params="k-means++",
            random_state=7,
        )
        mixture.fit(train_features, y_train)

        y_test_prob = mixture.predict_proba(test_features)[:,1]
        metrics += print_model_metrics(y_test, y_test_prob, confusion = confusion, verbose = False, return_metrics = True)

    metrics /=cv.get_n_splits()
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(*metrics))
    if return_f1:
        return metrics[0]
    return mixture

from sklearn.neighbors import KNeighborsClassifier
def run_knn_cv(features: DataFrame, y: DataFrame, cv, pipeline=None, nNeighbors=3, confusion = False, return_f1 = False, verbose = True):
    metrics = np.zeros(6)
    features = features.to_numpy()
    y = y.to_numpy()
    isBinary = True if np.unique(y).size == 2 else False
    for train, test in cv.split(features, y):
        train_features, test_features = features[train, :], features[test, :]
        y_train, y_test = y[train], y[test]

        if pipeline is not None:
            pipeline.fit(train_features)
            train_features = pipeline.transform(train_features)
            test_features = pipeline.transform(test_features)

        knn = KNeighborsClassifier(
            n_neighbors = nNeighbors, # since we have a very small dataset
            weights="distance",
            n_jobs=-1
        )
        knn.fit(train_features, y_train)

        y_test_prob = knn.predict_proba(test_features)[:,1] if isBinary else knn.predict_proba(test_features)
        metrics += print_model_metrics(y_test, y_test_prob, isBinary, confusion = confusion, verbose = False, return_metrics = True)

    metrics /=cv.get_n_splits()
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} | B_Acc: {:.3f}  \n'.format(*metrics))
    if return_f1:
        return metrics[0]
    return knn

from sklearn.tree import DecisionTreeClassifier
def run_tree_cv(features: DataFrame, y: DataFrame, cv, pipeline=None, max_depth=3, confusion = False, return_f1 = False, verbose = True):
    metrics = np.zeros(6)
    features = features.to_numpy()
    y = y.to_numpy()
    isBinary = True if np.unique(y).size == 2 else False
    for train, test in cv.split(features, y):
        train_features, test_features = features[train, :], features[test, :]
        y_train, y_test = y[train], y[test]

        if pipeline is not None:
            pipeline.fit(train_features)
            train_features = pipeline.transform(train_features)
            test_features = pipeline.transform(test_features)

        tree = DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=max_depth,
            max_features=None,
            random_state=7,
            class_weight="balanced",
        )
        tree.fit(train_features, y_train)

        y_test_prob = tree.predict_proba(test_features)[:,1] if isBinary else tree.predict_proba(test_features)
        metrics += print_model_metrics(y_test, y_test_prob, isBinary, confusion = confusion, verbose = False, return_metrics = True)

    metrics /=cv.get_n_splits()
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} | B_Acc: {:.3f}  \n'.format(*metrics))
    if return_f1:
        return metrics[0]
    return tree