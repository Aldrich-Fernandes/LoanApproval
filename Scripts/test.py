from math import sqrt
from random import gauss
import numpy as np
from DataHandle import DataMethod as DM


print("Hidden")
NoOfInputs = 11
NoOfNeurons = 7

# Xavier/Glorot weight initialization
scale = sqrt(2 / (NoOfInputs+NoOfNeurons))
weights = [[gauss(0, scale) for x in range(NoOfNeurons)] for y in range(NoOfInputs)]
print(f"Xavier:\n {weights}")
weights = [DM.Multiply(0.01, np.random.randn(1, NoOfNeurons).tolist()[0]) for i in range(NoOfInputs)]
print(f"Original:\n {weights}")

print("\n\nOutput")
NoOfInputs = 7
NoOfNeurons = 1

# Xavier/Glorot weight initialization
scale = sqrt(2 / (NoOfInputs+NoOfNeurons))
weights = [[gauss(1, scale) for x in range(NoOfNeurons)] for y in range(NoOfInputs)]
print(f"Xavier:\n {weights}")
weights = [DM.Multiply(0.01, np.random.randn(1, NoOfNeurons).tolist()[0]) for i in range(NoOfInputs)]
print(f"Original:\n {weights}")