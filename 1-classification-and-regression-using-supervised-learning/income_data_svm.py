import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn import cross_validation


inputFile = 'income_data.txt'

# Read the data
x = []
y = []
countClass1 = 0
countClass2 = 0
maxDataPoints = 25000

with open(inputFile, 'r') as f:
    for line in f.readlines():
        if countClass1 >= maxDataPoints and countClass2 >= maxDataPoints:
            break

        if '?' in line:
            continue

        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and countClass1 < maxDataPoints:
            x.append(data)
            countClass1 += 1

        if data[-1] == '>50K' and countClass2 < maxDataPoints:
            x.append(data)
            countClass2 += 1


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

# Create SVM classifier with a linear kernel
classifier = OneVsOneClassifier(LinearSVC(random_state=0))

# Train the classifier
classifier.fit(x, y)

# Cross validate using an 80/20 split for training and testing and predict the output for the training data
xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(x, y, test_size=0.2, random_state=5)
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(xTrain, yTrain)
yTestPrediction = classifier.predict(xTest)

# Compute the F1 score of the SVM classifier
f1 = cross_validation.cross_val_score(classifier, x, y, scoring='f1_weighted', cv=3)

print('F1 score: ' + str(round(100*f1.mean(), 2)) + '%')

# Predict output for a test datapoint
inputData = [
    '37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
    '0', '0', '40', 'United-States'
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

print(labelEncoder[-1].inverse_transform(predictedClass)[0])


