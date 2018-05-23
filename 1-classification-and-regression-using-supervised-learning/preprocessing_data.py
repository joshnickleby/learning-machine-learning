import numpy as np
from sklearn import preprocessing


input_data = np.array(
    [
        [5.1, -2.9, 3.3],
        [-1.2, 7.8, -6.1],
        [3.9, 0.4, 2.1],
        [7.3, -9.9, -4.5]
    ]
)


# Binarize data - Used to convert numerical values into boolean values threshold states that anything above that should
#                 return true; otherwise return false
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)

print("\nBinarized data:\n", data_binarized)


# Print mean and standard deviation - The mean & standard here are calculated from each column of the vector
#                                     See README for explanation on calculating this manually.
print("\nBEFORE:")
print("Mean =", input_data.mean(axis=0))
print("Std dev =", input_data.std(axis=0))


# TODO: How is preprocessing scaling it?
# Remove mean - It remains useful to remove the mean from our feature vector so that each feature has a basis around
#               zero effectively removing bias.
data_scaled = preprocessing.scale(input_data)

print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0))
print("Std dev =", data_scaled.std(axis=0))


# TODO: How is preprocessing figuring out the min max?
# Min max scaling - Basically reformats the original data so that they are between 0 and 1 respectively to their values
#                   within the given set
data_scalar_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scalar_minmax.fit_transform(input_data)

print("\nMin max scaled data:\n", data_scaled_minmax)


# TODO: How is preprocessing normalizing it?
# Normalize data
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')

print("\nL1 normalized data:\n", data_normalized_l1)
print("\nL2 normalized data:\n", data_normalized_l2)
