import numpy as np

X_train = np.random.rand(100, 6)  # Example input data
y_train = np.random.randint(0, 2, size=(100, 1))

for epoch in range(50):
     for i in range(0, 100, 32):
          # Extract a batch
          print(len(X_train[i:i+32]))
     input()