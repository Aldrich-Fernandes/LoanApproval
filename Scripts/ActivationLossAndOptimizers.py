import numpy as np
from math import log

from DataHandle import DataMethod as DM

#Activations
class ReLU:
    def forward(self, inputs):
        self.outputs = [[0 if element < 0 else element for element in entry]
                        for entry in inputs]

    def backward(self, dvalues):
        self.dinputs = [[1 if element < 0 else 1 for element in entry] 
                        for entry in dvalues]

class Sigmoid:
    def forward(self, inputs):
        inputs = list(map(lambda z: z[0], inputs))
        self.outputs = [1 / (np.exp(-val) + 1) for val in inputs]
    
    def backward(self, dvalues):
        self.dinputs = [[(np.exp(-x) / ((1 + 2*np.exp(-x) + np.exp(-x*2))))] for x in dvalues]

# Loss
class Loss:
    def calculate(self, output, y):
        SampleLosses = self.forward(output, y)
        return SampleLosses #DataLoss

class BinaryCrossEntropy(Loss): 
    def forward(self, predictions, TrueVals):
        # Remove any 0s or 1s to avoid arithmethic errors
        for index, val in enumerate(predictions):
            if val < 0.0000001:
                predictions[index] = 0.0000001
            elif val > 0.9999999: 
                predictions[index] = 0.9999999

        SampleLoss = [-(val1+val2) for val1, val2 in zip(DM.Multiply(TrueVals, [log(x) for x in predictions]),  #Probabilty of 1
                                                        DM.Multiply([1-x for x in TrueVals], [log(1-x) for x in predictions]))] # Probablity of 0
        
        SampleLoss = round(sum(SampleLoss) / len(SampleLoss), 24)

        return SampleLoss
    
    def backward(self, dvalues, TrueVals):

        for index, val in enumerate(dvalues):
            if val < 0.0000001:
                dvalues[index] = 0.0000001
            elif val > 0.9999999:
                dvalues[index] = 0.9999999

        # -(true / dvalues) + ( (1-true) / (1-dvalues))
        vals1 = [-(x/y) for x, y in zip(TrueVals, dvalues)]
        vals2 = [x/y for x, y in zip([1-i for i in TrueVals],
                                     [1-j for j in dvalues])]
        self.dinputs = [v1+v2 for v1, v2 in zip(vals1, vals2)]
    