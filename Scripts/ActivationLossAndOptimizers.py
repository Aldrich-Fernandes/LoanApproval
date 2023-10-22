import numpy as np
from math import log
from abc import ABC, abstractmethod

from DataHandle import DataMethod as DM

#Activations
class Activation(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass
        
    @abstractmethod
    def backward(self, dvalues):
        pass
    
class ReLU(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = [[0 if element < 0 else element for element in entry]
                        for entry in inputs]

    def backward(self, dvalues):
        dvals = dvalues.copy()
        self.dinputs = [[0 if element <= 0 else dvals[entryIndex][elementIndex] for elementIndex, element in enumerate(entry)]
                        for entryIndex, entry in enumerate(self.inputs)]

class Sigmoid(Activation):
    def forward(self, inputs):
        inputs = [z[0] for z in inputs]
        self.outputs = [1 / (np.exp(-val) + 1) for val in inputs]
    
    def backward(self, dvalues):
        
        self.dinputs = [[a*b*c] for a,b,c in zip(dvalues, [1-i for i in self.outputs], self.outputs)]

# Loss
class BinaryCrossEntropy: 
    def forward(self, predictions, TrueVals):
        # Remove any 0s or 1s to avoid arithmethic errors
        predictions = clipEdges(predictions)

        SampleLoss = [-(val1+val2) for val1, val2 in zip(DM.Multiply(TrueVals, [log(x) for x in predictions]),  #Probabilty of 1
                                                        DM.Multiply([1-x for x in TrueVals], [log(1-x) for x in predictions]))] # Probablity of 0
        
        SampleLoss = round(sum(SampleLoss) / len(SampleLoss), 24)

        return SampleLoss
    
    def backward(self, dvalues, TrueVals):

        dvalues = clipEdges(dvalues)

        # -(true / dvalues) + ( (1-true) / (1-dvalues))
        vals1 = [-(x/y) for x, y in zip(TrueVals, dvalues)]
        vals2 = [x/y for x, y in zip([1-i for i in TrueVals], [1-j for j in dvalues])]
        self.dinputs = [v1+v2 for v1, v2 in zip(vals1, vals2)]
    
class OptimizerSGD:
    def __init__(self, learningRate=0.0075): # learning rate too high = no learning
        self.__LearningRate = learningRate

    def UpdateParameters(self, layer): ### Issue - result keeps increasing to 1 until crash
        AdjustedDWeight = [DM.Multiply(-self.__LearningRate, sample) for sample in layer.dweights]
        layer.weights = [[a+b for a, b in zip(layer.weights[x], AdjustedDWeight[x])] for x in range(len(layer.weights))]

        layer.biases = [a+b for a,b in zip(layer.biases, DM.Multiply(-self.__LearningRate, layer.dbiases))]

def clipEdges(list):
    for index, val in enumerate(list):
        if val < 0.0000001:
            list[index] = 0.0000001
        elif val > 0.9999999:
            list[index] = 0.9999999
    return list