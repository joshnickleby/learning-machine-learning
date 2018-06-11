from sklearn import preprocessing

# Sample input labels
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']


# Create label encoder and fit the labels - As far as I can tell, it takes the input labels and assigns a number to each
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# Print the mapping
print("Label mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '→', i)

# Encode a set of labels using the encoder - Checking if the encoder is correctly converting them
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)


print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))

# Decode a set of values using the encoder
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)

print("\nEncoded values =", encoded_values)
print("Decoded labels =", list(decoded_list))
