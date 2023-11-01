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
        self.outputs = [[max(0, element) for element in entry]
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
        
        self.dinputs = [[a*b*(1-b)] for a,b in zip(dvalues, self.outputs)]

# Loss
class BinaryCrossEntropy: 
    def forward(self, predictions, TrueVals):
        # Remove any 0s or 1s to avoid arithmethic errors
        predictions = clipEdges(predictions)

        # [-(tv * log(p) + (1 - tv) * log(1 - p)) for tv, p in zip(true_vals, predictions)]
        SampleLoss = [-(tVal * log(pVal) + (1- tVal) * log(1-pVal)) for tVal, pVal in zip(TrueVals, predictions)]
        
        SampleLoss = round(sum(SampleLoss) / len(SampleLoss), 24)

        return SampleLoss
    
    def backward(self, dvalues, TrueVals):

        dvalues = clipEdges(dvalues)

        # -(true / dvalues) + ( (1-true) / (1-dvalues))
        self.dinputs = [-(x/y) + (1-x)/(1-y) for x, y in zip(TrueVals, dvalues)]
    

class OptimizerSGD:
    def __init__(self, InitialLearningRate=0.1, decay=0.05): # learning rate too high = no learning
        self.InitailLearningRate = InitialLearningRate
        self.decay = decay

        self.activeLearningRate = InitialLearningRate

    def adjustLearningRate(self, iter):
        self.activeLearningRate = self.InitailLearningRate / (1 + self.decay * iter)

    def UpdateParameters(self, layer): ### Issue - result keeps increasing to 1 until crash
        AdjustedDWeight = [DM.Multiply(-self.activeLearningRate, sample) for sample in layer.dweights]
        layer.weights = [[a+b for a, b in zip(layer.weights[x], AdjustedDWeight[x])] for x in range(len(layer.weights))]

        layer.biases = [a+b for a,b in zip(layer.biases, DM.Multiply(-self.activeLearningRate, layer.dbiases))]

def clipEdges(list):
    for index, val in enumerate(list):
        if val < 1e-10:
            list[index] = 1e-10
        elif val > 1 - (1e-10):
            list[index] = 1 - (1e-10)
    return list