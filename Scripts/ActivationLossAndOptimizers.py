import numpy as np
from math import log

from DataHandle import DataMethod as DM

#Activations
class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = [[0 if element < 0 else element for element in entry]
                        for entry in inputs]

    def backward(self, dvalues):
        dvals = dvalues.copy()
        self.dinputs = [[0 if element <= 0 else dvals[entryIndex][elementIndex] for elementIndex, element in enumerate(entry)]
                        for entryIndex, entry in enumerate(self.inputs)]

class Sigmoid:
    def forward(self, inputs):
        inputs = list(map(lambda z: z[0], inputs))
        self.outputs = [1 / (np.exp(-val) + 1) for val in inputs]
    
    def backward(self, dvalues):
        
        self.dinputs = [[a*b*c] for a,b,c in zip(dvalues, [1-i for i in self.outputs], self.outputs)]

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
    

class OptimizerSGD:
    def __init__(self, learningRate=0.01):
        self.__learningRate = learningRate

    def UpdateParameters(self, layer): ### Issue - result keeps increasing to 1 until crash
        AdjustedDWeight = [DM.Multiply(-self.__learningRate, sample) for sample in layer.dweights]
        layer.weights = [[a+b for a, b in zip(layer.weights[x], AdjustedDWeight[x])] for x in range(len(layer.weights))]

        layer.biases = [a+b for a,b in zip(layer.biases, DM.Multiply(-self.__learningRate, layer.dbiases))]