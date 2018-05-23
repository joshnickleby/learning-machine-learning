
## Information

Section: Classification and Regression Using Supervised Learning


## Important terms

**Preprocessing**

**Binarization**

**Mean removal**

**Scaling**

**Label encoding**

**Logistic regression classifier**


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
