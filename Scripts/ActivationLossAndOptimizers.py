import numpy as np
from math import log

from DataHandle import DataMethod as DM

#Activations
class ReLU:
    def forward(self, inputs):
        for rowIndex, entry in enumerate(inputs):
            for index, element in enumerate(entry):
                if element < 0:
                    inputs[rowIndex][index] = 0
        return inputs

class Sigmoid:
    def forward(self, inputs):
        inputs = list(map(lambda z: z[0], inputs))
        return [1 / (np.exp(-val) + 1) for val in inputs]

# Loss
class Loss:
    def calculate(self, output, y):
        SampleLosses = self.forward(output, y)
        return SampleLosses #DataLoss

class BinaryCrossEntropy(Loss): 
    def forward(self, predictions, TrueVals):
        # Remove any 0s or 1s to avoid arithmethic errors
        for index, val in enumerate(predictions):
            if val < 1e-7:
                predictions[index] = 0.0000001
            elif val > 1- 1e-7:
                predictions[index] = 0.9999999

        SampleLoss = [-(val1+val2) for val1, val2 in zip(DM.Multiply(TrueVals, [log(x) for x in predictions]),  #Probabilty of 1
                                                        DM.Multiply([1-x for x in TrueVals], [log(1-x) for x in predictions]))] # Probablity of 0
        
        SampleLoss = round(sum(SampleLoss) / len(SampleLoss), 16)

        return SampleLoss
    

# Optimizers
