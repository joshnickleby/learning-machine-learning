import numpy as np
from sklearn import preprocessing
from utils.list import List


def printInfo(label, data):
    print(f'\n{label}:')
    print("Mean =", data.mean(axis=0))
    print("Std dev =", data.std(axis=0))
    

def normalizeData(normalizingType):
    normalizedData = preprocessing.normalize(inputData, norm=normalizingType)

    print(f'\n{normalizingType.capitalize()} normalized data:\n', normalizedData)


inputData = np.array(
    [
        [ 5.1, -2.9,  3.3],
        [-1.2,  7.8, -6.1],
        [ 3.9,  0.4,  2.1],
        [ 7.3, -9.9, -4.5]
    ]
)

# Binarize data - Used to convert numerical values into boolean values threshold states that anything above that should
#                 return true; otherwise return false
dataBinarized = preprocessing.Binarizer(threshold=2.1).transform(inputData)

print("\nBinarized data:\n", dataBinarized)


# Print mean and standard deviation - The mean & standard here are calculated from each column of the vector
#                                     See README for explanation on calculating this manually.
printInfo('BEFORE', inputData)


# TODO: How is preprocessing scaling it?
# Remove mean - It remains useful to remove the mean from our feature vector so that each feature has a basis around
#               zero effectively removing bias.
dataScaled = preprocessing.scale(inputData)

printInfo('AFTER', dataScaled)


# TODO: How is preprocessing figuring out the min max?
# Min max scaling - Basically reformats the original data so that they are between 0 and 1 respectively to their values
#                   within the given set
dataScalarMinMax = preprocessing.MinMaxScaler(feature_range=(0,1))
dataScaledMinMax = dataScalarMinMax.fit_transform(inputData)

print("\nMin max scaled data:\n", dataScaledMinMax)


# TODO: How is preprocessing normalizing it?
# Normalize data
List('l1', 'l2').map(normalizeData)
