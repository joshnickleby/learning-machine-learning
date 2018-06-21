import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Define sample labels for the ground truth and the predicted output
trueLabels = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
predictedLabels = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]

# Create confusion matrix
confusionMatrix = confusion_matrix(trueLabels, predictedLabels)

# Visualize the confusion matrix
#       The top left to bottom right diagonal should show white while all the rest should show black (100% accurate)
#       In the above array, if the true matches the prediction it stays white but once a predicted misclassifies
#       based on the true then it becomes darker
plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()

ticks = np.arange(5)

plt.yticks(ticks, ticks)
plt.xticks(ticks, ticks)

plt.ylabel('True labels')
plt.xlabel('Predicted labels')

plt.show()

# Print the classification report
targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4']
print('\n', classification_report(trueLabels, predictedLabels, target_names=targets))

# OUTPUTS
#   recall = ?
#   f1-score = ?
#   support = number of occurrences in the true dataset
#
#               precision    recall  f1-score   support
#     Class-0       1.00      0.67      0.80         3
#     Class-1       0.33      1.00      0.50         1
#     Class-2       1.00      1.00      1.00         2
#     Class-3       0.67      0.67      0.67         3
#     Class-4       1.00      0.50      0.67         2
# avg / total       0.85      0.73      0.75        11
