
## Information ##

I am going through Packt's - Artificial Intelligence with Python

Left off: Label encoding


## Refresher Math ##

### Process to find the Standard Deviation ###

reference: preprocessing_data.py

    Dataset = [100, 50, 36, 12, 45]

#### Mean ####

The average of this set:

    (100 + 50 + 36 + 12 + 45) / 5 = 48.6
    243 / 5 = 48.6
    
    
#### Variance ####

1. The difference of each datum and the mean are found
2. Each of these values area squared
3. Find the average of these 5 new values


    ((100 - 48.6)² + (50 - 48.6)² + (36 - 48.6)² + (12 - 48.6)² + (45 - 48.6)²) / 5 = 894.52
    (54.4² + 1.4² + (-12.6)² + (-36.6)² + (-3.6)²) / 5 = 894.52
    (2959.36 + 1.96 + 158.76 + 1339.56 + 12.96) / 5 = 894.52
    4472.6 / 5 = 894.52


#### Standard Deviation ####

The square root of the variance which delineates the standard variance based off the mean.

     √894.52 = 29.9085
     √894.52 = ~30



### TO LEARN ###

* Reason and means for the process of "Removing the mean"
* Scalar Min Max
* L1 Normalization
* L2 Normalization


## Glossary ##

*Normalization*

*Least Absolute Normalization (L1 Normalization)*

Makes sure that the sum of absolute values equals 1 in each row. This is better for datasets where you DON'T care about
the outliers.

*Least Squared Normalization (L2 Normalization)*

Makes sure that the sum of squared values equals 1 in each row. This is better for datasets where you DO care about
the outliers.