import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn import cross_validation


class FileClassifier:
    file = ''
    countClass1, countClass2 = 0, 0
    maxDataPoints = 0
    x = []

    def __init__(self, file, maxDataPoints):
        self.file = file
        self.maxDataPoints = maxDataPoints

    def extractData(self):
        with open(self.file, 'r') as f:
            for line in f.readlines():
                if self.gteqMax(self.countClass1) and self.gteqMax(self.countClass2):
                    break

                if '?' in line:
                    continue

                data = line[:-1].split(', ')

                self.classifyDataPoint(data, '<=50K', self.countClass1)
                self.classifyDataPoint(data, '>50K', self.countClass2)

        return self.x

    def classifyDataPoint(self, data, criterion, classCount):
        if data[-1] == criterion and classCount < self.maxDataPoints:
            self.x.append(data)
            classCount += 1

    def gteqMax(self, classCount):
        return classCount >= self.maxDataPoints


def classifyData(x, y):
    classifier = OneVsOneClassifier(LinearSVC(random_state=0))
    classifier.fit(x, y)
    return classifier


# Start the routine
fileClassifier = FileClassifier(file='income_data.txt', maxDataPoints=25000)

# Read the data
x = fileClassifier.extractData()
y = []

# Convert to a numpy array
x = np.array(x)

# Convert string data to numerical data
labelEncoder = []
xEncoded = np.empty(x.shape)

for i, item in enumerate(x[0]):
    if item.isdigit():
        xEncoded[:, i] = x[:, i]
    else:
        labelEncoder.append(preprocessing.LabelEncoder())
        xEncoded[:, i] = labelEncoder[-1].fit_transform(x[:, i])

x = xEncoded[:, :-1].astype(int)
y = xEncoded[:, -1].astype(int)

# Create SVM classifier with a linear kernel and train
classifier = classifyData(x, y)

# Cross validate using an 80/20 split for training and testing and predict the output for the training data
xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(x, y, test_size=0.2, random_state=5)
classifier = classifyData(xTrain, yTrain)
yTestPrediction = classifier.predict(xTest)

# Compute the F1 score of the SVM classifier
f1 = cross_validation.cross_val_score(classifier, x, y, scoring='f1_weighted', cv=3)

f1Score = str(round(100*f1.mean(), 2))

print(f'F1 score: {f1Score}%')

# Predict output for a test datapoint
inputData = [
    '37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners',
    'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States'
]

# Encode test datapoint
inputDataEncoded = [-1] * len(inputData)
count = 0

for i, item in enumerate(inputData):
    if item.isdigit():
        inputDataEncoded[i] = int(inputData[i])
    else:
        inputDataEncoded[i] = int(labelEncoder[count].transform([inputData[i]]))
        count += 1

inputDataEncoded = np.array(inputDataEncoded).reshape(1, -1)

# Run classifier on encoded datapoint and print output
predictedClass = classifier.predict(inputDataEncoded)

testDataPointOut = labelEncoder[-1].inverse_transform(predictedClass)[0]

print(testDataPointOut)


# Test expected
expectedF1Score = '70.82' == f1Score
expectedTestDataPointOut = '<=50K' == testDataPointOut

print(f'Correct F1 score: {expectedF1Score}')
print(f'Correct test data point output: {expectedTestDataPointOut}')