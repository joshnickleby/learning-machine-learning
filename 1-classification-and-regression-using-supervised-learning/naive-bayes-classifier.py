import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

from utils.utilities import visualize_classifier


# Convience methods (train = training data)
def trainClassifier(xTrain, yTrain, xTest):
    # Create classifier
    classifier = GaussianNB()

    # Train the classifier
    classifier.fit(xTrain, yTrain)

    # Predict the values for training data
    yPrediction = classifier.predict(xTest)

    return classifier, yPrediction


def visualize(classifier, yTest, yTestPrediction, xTest, descriptor):
    # Compute accuracy
    accuracy = 100.0 * (yTest == yTestPrediction).sum() / xTest.shape[0]
    print(f'Accuracy of {descriptor} classifier = {round(accuracy, 2)}%')

    # Visualize the performance of the classifier
    visualize_classifier(classifier, xTest, yTest)


def checkValues(classifier, x, y, scoring):
    numFolds = 3
    values = cross_validation.cross_val_score(classifier, x, y, scoring=scoring, cv=numFolds)

    label = scoring.split("_")[0].capitalize()

    print(f'{label}: {str(round(100 * values.mean(), 2))}%')
    return values




# Load data from input file
inputFile = 'data_multivar_nb.txt'
data = np.loadtxt(inputFile, delimiter=",")
x, y = data[:, :-1], data[:, -1]


# Train a Naive Bayes classifier
classifier, yPrediction = trainClassifier(x, y, x)


# Compute accuracy
visualize(classifier, y, yPrediction, x, 'a Naive Bayes')


# Split data into training and test data
xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(x, y, test_size=0.2, random_state=3)


# Train the new classifier
newClassifier, yTestPrediction = trainClassifier(xTrain, yTrain, xTest)


# Compute accuracy of the classifier
visualize(newClassifier, yTest, yTestPrediction, xTest, 'the new')


# Check accuracy, precision, recall, and f1
tests = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
testResults = [checkValues(classifier, x, y, test) for test in tests]
