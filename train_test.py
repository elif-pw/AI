import h5py
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import warnings

warnings.simplefilter("ignore")  # warning suppression

h5Data = "output/data.h5"
h5Labels = "output/labels.h5"
h5Results = "output/results.h5"

h5fData = h5py.File(h5Data, "r")
h5FLabel = h5py.File(h5Labels, "r")
h5fResults = h5py.File(h5Results, "w")

globalFeaturesString = h5fData["globalFeatures"]
globalFeatures = np.array(globalFeaturesString)
globalLabelsString = h5FLabel["labels"]
globalLabels = np.array(globalLabelsString)

h5fData.close()
h5FLabel.close()

nSplits = 10
treeNumber = 100
testSize = 0.1
seed = 9
scoring = {"acc": "accuracy", "prec": "precision"}

models = []
models.append(("RFC", RandomForestClassifier(
    n_estimators=treeNumber, random_state=seed)))
models.append(("SVC", SVC(random_state=seed)))
models.append(("KNN", KNeighborsClassifier()))
models.append(("LR", LogisticRegression(random_state=seed)))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("NB", GaussianNB()))
models.append(("CART", DecisionTreeClassifier(random_state=seed)))
models.append(("ABC", AdaBoostClassifier()))
models.append(("QDA", QuadraticDiscriminantAnalysis()))

(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(
    globalFeatures, globalLabels, test_size=testSize, random_state=seed)

# results = []
for name, model in models:
    kfold = KFold(n_splits=nSplits, random_state=seed)
    cvResult = cross_validate(model, trainDataGlobal, trainLabelsGlobal,
                              cv=kfold, scoring=scoring, return_train_score=True)
    # results.append((name, cvResult["test_acc"], cvResult["test_prec"]))
    h5fData.create_dataset(name+"_acc", data=np.array(cvResult["test_acc"]))
    h5fData.create_dataset(name+"_prec", data=np.array(cvResult["test_prec"]))
    h5fData.create_dataset(
        name+"_fit_time", data=np.array(cvResult["fit_time"]))
    h5fData.create_dataset(
        name+"_score_time", data=np.array(cvResult["score_time"]))
