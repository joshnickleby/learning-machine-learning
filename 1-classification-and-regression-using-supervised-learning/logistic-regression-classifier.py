import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

from utils.utilities import visualize_classifier

# Define same input data
X = np.array([
    [3.1, 7.2],
    [4.0, 6.7],
    [2.9, 8.0],
    [5.1, 4.5],
    [6.0, 5.0],
    [5.6, 5.0],
    [3.3, 0.4],
    [3.9, 0.9],
    [2.8, 1.0],
    [0.5, 3.4],
    [1.0, 4.0],
    [0.6, 4.9]
])

y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])


# todo: What is this doing?
# Create a logistic regression classifier - (change C from 1 to 100 to see accuracy change) This imposes a certain
#                                            penalty on misclassification
classifier = linear_model.LogisticRegression(solver='liblinear', C=100)


# todo: What is this doing?
# Train the classifier
classifier.fit(X, y)


# Visualize the performance of the classifier - see utils/utilities.py
visualize_classifier(classifier, X, y)
