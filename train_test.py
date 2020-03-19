import h5py
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
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

treeNumber = 100
testSize = 0.1
seed = 9
scoring = "accuracy"

modelRFC = RandomForestClassifier(n_estimators=treeNumber, random_state=seed)
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(
    np.array(globalFeatures), np.array(globalLabels), test_size=testSize, random_state=seed)

kfold = KFold(n_splits=10, random_state=seed)
cvScoreRFC = cross_val_score(
    modelRFC, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
print("Rand Forest Classifier score: {}".format(cvScoreRFC))

modelSVC = SVC(random_state=seed)

kfold = KFold(n_splits=10, random_state=seed)
cvScoreSVC = cross_val_score(
    modelSVC, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
print("SVC score: {}".format(cvScoreSVC))
