import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sns
import numpy as np
sns.set_palette("muted")
    

def calc_f1(p_and_r):
    p, r = p_and_r
    if p == 0 and r == 0:
        return 0
    return (2*p*r)/(p+r)

from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
def auc_and_f1(y, y_prob, **kwargs):

    precision, recall, threshold = precision_recall_curve(y, y_prob, pos_label=1)

    #Find the threshold value that gives the best F1 Score
    best_f1_index = np.argmax([ calc_f1(p_r) for p_r in zip(precision, recall)])
    best_threshold, best_precision, best_recall = threshold[best_f1_index], precision[best_f1_index], recall[best_f1_index]

    # Calulcate predictions based on the threshold value
    y_pred = np.where(y_prob > best_threshold, 1, 0)

    # Calculate metrics
    roc= roc_auc_score(y, y_prob)
    f1 = f1_score(y, y_pred, pos_label=1)
    return roc + f1


# Print the F1, Precision, Recall, ROC-AUC, and Accuracy Metrics 
# Since we are optimizing for F1 score - we will first calculate precision and recall and 
# then find the probability threshold value that gives us the best F1 score

def print_model_metrics(y_test, y_test_prob, confusion = False, verbose = True, return_metrics = False):

    precision, recall, threshold = precision_recall_curve(y_test, y_test_prob, pos_label = 1)

    #Find the threshold value that gives the best F1 Score
    best_f1_index =np.argmax([calc_f1(p_r) for p_r in zip(precision, recall)])
    best_threshold, best_precision, best_recall = threshold[best_f1_index], precision[best_f1_index], recall[best_f1_index]
    
    # Calulcate predictions based on the threshold value
    y_test_pred = np.where(y_test_prob > best_threshold, 1, 0)
    
    # Calculate all metrics
    if np.unique(y_test).size > 2:
        averageF1 = "weighted"
    else:
        averageF1 = "binary"
    f1 = f1_score(y_test, y_test_pred, pos_label = 1, average = averageF1)
    roc_auc = roc_auc_score(y_test, y_test_prob, multi_class="ovo")
    acc = accuracy_score(y_test, y_test_pred)
    
    
    if confusion:
        # Calculate and Display the confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)

        plt.title('Confusion Matrix')
        sns.set(font_scale=1.0) #for label size
        sns.heatmap(cm, annot = True, fmt = 'd', xticklabels = ['No Clickbait', 'Clickbait'], yticklabels = ['No Clickbait', 'Clickbait'], annot_kws={"size": 14}, cmap = 'Blues')# font size

        plt.xlabel('Truth')
        plt.ylabel('Prediction')
        
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(f1, best_precision, best_recall, roc_auc, acc))
    
    if return_metrics:
        return np.array([f1, best_precision, best_recall, roc_auc, acc])
    
    

# Run Simple Log Reg Model and Print metrics
from sklearn.linear_model import SGDClassifier, LogisticRegression

from pandas import DataFrame

def run_log_reg_cv(features: DataFrame, y: DataFrame, cv, pipeline=None, C = 1, penalty="l2", confusion = False, return_f1 = False, verbose = True):
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

        log_reg = LogisticRegression(
            penalty=penalty,
            dual=True,
            C = C,
            class_weight="balanced",
            random_state=7,
            solver="liblinear",
            max_iter=100000,
        )
        log_reg.fit(train_features, y_train)

        y_test_prob = log_reg.predict_proba(test_features)[:,1]
        metrics += print_model_metrics(y_test, y_test_prob, confusion = confusion, verbose = False, return_metrics = True)

    metrics /=cv.get_n_splits()
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(*metrics))
    if return_f1:
        return metrics[0]
    return log_reg

from sklearn.svm import LinearSVC

def run_svm_cv(features: DataFrame, y: DataFrame, cv, pipeline=None, C = 1, penalty="l2", confusion = False, return_f1 = False, verbose = True):
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

        svm = LinearSVC(
            penalty=penalty,
            loss="hinge",
            dual=True,
            C = C,
            class_weight="balanced",
            random_state=7,
            max_iter=100000,
        )
        svm.fit(train_features, y_train)

        y_test_scores = svm.decision_function(test_features)
        metrics += print_model_metrics(y_test, y_test_scores, confusion = confusion, verbose = False, return_metrics = True)

    metrics /=cv.get_n_splits()
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(*metrics))
    if return_f1:
        return metrics[0]
    return svm

from sklearn.naive_bayes import GaussianNB

def run_gaussian_cv(features: DataFrame, y: DataFrame, cv, pipeline=None, confusion = False, return_f1 = False, verbose = True):
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

        gaussian = GaussianNB()
        gaussian.fit(train_features, y_train)

        y_test_prob = gaussian.predict_proba(test_features)
        metrics += print_model_metrics(y_test, y_test_prob, confusion = confusion, verbose = False, return_metrics = True)

    metrics /=cv.get_n_splits()
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(*metrics))
    if return_f1:
        return metrics[0]
    return gaussian