import numpy as np
import os
import h5py
import mahotas
import cv2 as cv
from sklearn.preprocessing import LabelEncoder
from skimage import feature

size = tuple((100, 100))

# shapes
def fd_hu_moments(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    feat = cv.HuMoments(cv.moments(img)).flatten()
    return feat

# haralick texture
def fd_haralick(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# color histogram
def fd_histogram(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([img], [0, 1, 2], None, [
        8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv.normalize(hist, hist)
    return hist.flatten()

# local features
def fd_localBinaryPatters(image, numPoints=24, radius=8):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(
        gray, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def getGlobalFeatures(file):
    image = cv.imread(file)
    image = cv.resize(image, size)
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)
    fv_lbp = fd_localBinaryPatters(image)
    # don't forget to add feature vector (fv_) to the hstack here
    return np.hstack([fv_hu_moments, fv_haralick, fv_histogram, fv_lbp])


trainPath = "data/Training"
h5Data = "output/data.h5"
h5Labels = "output/labels.h5"

trainLabels = os.listdir(trainPath)

globalFeatures = []
labels = []

for trainName in trainLabels:
    dir = os.path.join(trainPath, trainName)

    for file in os.listdir(dir):
        globalFeature = getGlobalFeatures(dir+"/"+file)

        labels.append(trainName)
        globalFeatures.append(globalFeature)
    print("{} class processed".format(trainName))

targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)

h5fData = h5py.File(h5Data, "w")
h5fData.create_dataset("globalFeatures", data=np.array(globalFeatures))
h5fData.close()

h5fLabel = h5py.File(h5Labels, "w")
h5fLabel.create_dataset("labels", data=np.array(target))
h5fLabel.close()

print("\nFeatures extracted")
