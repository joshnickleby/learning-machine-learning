
## Information

Section: Classification and Regression Using Supervised Learning


## Important terms

**Binarization**

**Label encoding**

**Mean removal**

**Naïve Bayes**

A technique to build classifiers with [Bayes theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem). The theorem
describes the probability of an event occurring based on different conditions relating to the event. The naïve part
is based on the independence assumption, which states that the value of any given feature will remain independent
of the value of any other feature.

_Example_

> We may classify a living tree as having bark, branches, leaves. The classifier then considers each feature
> independently and aggregates it all as a probability that we can classify it as a living tree (maybe vs a dead one).


**Preprocessing**

**Scaling**

**Logistic regression classifier**

**[Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)**

A figure or a table used to describe the performance of a classifier. It is usually determined from a test dataset
which has a known ground truth (eg ??). The matrix compares each class with the others to determine how many
samples the classifier has misclassified.

Binary classification confusion matrix example (output either equals 0 or 1):

* **True positives**: Samples which we predicted 1 for the output and the ground truth equals 1.
* **True negatives**: Samples which we predicted 0 for the output and the ground truth equals 0.
* **False positives**: Samples which we predicted 1 for the output and the ground truth equals 0. _(Type I error)_
* **False negatives**: Samples which we predicted 0 for the output and the ground truth equals 1. _(Type II error)_

**Support Vector Machine (SVM)**

A classifier that we define using a separate hyperplane between the classes.

## Refresher math

### Process to find the Standard Deviation

reference: preprocessing_data.py

    Dataset = [100, 50, 36, 12, 45]

#### Mean

The average of this set:

    (100 + 50 + 36 + 12 + 45) / 5 = 48.6
    243 / 5 = 48.6
    
    
#### Variance

1. The difference of each datum and the mean are found
2. Each of these values area squared
3. Find the average of these 5 new values

```
((100 - 48.6)² + (50 - 48.6)² + (36 - 48.6)² + (12 - 48.6)² + (45 - 48.6)²) / 5 = 894.52
(54.4² + 1.4² + (-12.6)² + (-36.6)² + (-3.6)²) / 5 = 894.52
(2959.36 + 1.96 + 158.76 + 1339.56 + 12.96) / 5 = 894.52
4472.6 / 5 = 894.52
```

#### Standard Deviation

The square root of the variance which delineates the standard variance based off the mean.

     √894.52 = 29.9085
     √894.52 = ~30
