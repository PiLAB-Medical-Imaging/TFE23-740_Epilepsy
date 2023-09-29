import json
import utils

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

df = utils.getReducedDS()
X, y, y3 = utils.splitFeatureLabels(df)
X_train, X_test, y_train, y_test, y3_train, y3_test = utils.splitTrainTestDF(X, y, y3, 0.4)

mad_outliers = utils.MADOutlierRemotion(3)
mad_outliers.fit(X_train)

X_train = mad_outliers.transform(X_train)
X_test = mad_outliers.transform(X_test)

for name, algorithm in [
    ("logreg", LogisticRegression(C=1e-6, random_state=7, max_iter=10000, class_weight="balanced")),
    ("svm", SVC(C=1e-6, random_state=7, class_weight="balanced")),
    ("knn", CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=3), cv=3)),
    ("mlp", MLPClassifier(random_state=7, alpha=2, learning_rate="adaptive", max_iter=1000)),
    ("gauss", GaussianNB())
]:

    res = {}

    multiIndices = utils.countByFeatureGroups(X_train.iloc[:, 5:]).index

    region_prec = ""
    image_type_prec = ""
    feature_name_prec = ""
    X_curr = X_train.copy()

    for region, image_type, feature_name in tqdm(multiIndices):

        if region != region_prec:
            X_curr = X_curr.filter(regex=region)
            if X_curr.shape[1] == 0:
                X_curr = X_train.filter(regex=region)

        if image_type != image_type_prec:
            X_curr = X_curr.filter(regex=image_type)
            if X_curr.shape[1] == 0:
                X_curr = X_train.filter(regex=image_type)

        # print(region+"_"+image_type+"_"+feature_name)

        res[region+"_"+image_type+"_"+feature_name] = utils.scoreCV(
            algorithm,
            X_curr, y_train,
            regex = region+"_.*_"+image_type+"_.*_"+feature_name,
            decision=True if name == "svm" else False,
            doPrints=False
        )

        region_prec = region
        image_type_prec = image_type

    with open(f"../study/stats/results-{name}-fix40-3cv.json", "w") as outfile:
        json.dump(res, outfile, indent=2, sort_keys=True)