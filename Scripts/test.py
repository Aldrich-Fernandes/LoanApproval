from DataHandle import DataMethod
import numpy as np

a = np.random.randn(11, 7)
b = np.random.randn(1, 11)

DataMethod.DotProduct(a, DataMethod.Transpose(b))