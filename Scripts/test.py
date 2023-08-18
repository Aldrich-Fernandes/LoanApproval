import numpy as np
val = [0, 1, 0.4, -12, -4.65, -0.23]

for x in val:
        print(x)
        print(1 / (1 + np.exp(-x)))
        print(np.exp(x) / (1 + np.exp(x)))
        print("\n")