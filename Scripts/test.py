# Import necessary libraries (only built-in Python libraries are used)
import numpy as np

# Sample data (replace this with your dataset)
X_train = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]])

# Calculate the mean and standard deviation for each feature
mean = np.mean(X_train, axis=0)
std_dev = np.std(X_train, axis=0)
print(mean)
print(std_dev)
# Standardize the data (Z-score scaling)
X_train_scaled = (X_train - mean) / std_dev

# Print the scaled data
print("Scaled Data:")
print(X_train_scaled)
