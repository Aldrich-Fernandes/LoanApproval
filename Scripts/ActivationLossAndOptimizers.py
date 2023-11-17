from math import exp, log
from abc import ABC, abstractmethod

from DataHandle import DataMethod as DM

#Activations
class Activation(ABC): # Scales 
    @abstractmethod
    def forward(self, inputs): # Formula
        pass

    @abstractmethod
    def backward(self, dvalues): # Diretive of the Formula
        pass

class ReLU(Activation): # Rectified Linear Unit
    def __init__(self):
        self.__ID = "ReLU"

    def getID(self):
        return self.__ID

    def forward(self, inputs): # limits between (>= 0)
        self.inputs = inputs
        self.outputs = [[max(0, element) for element in entry]
                        for entry in inputs]

    def backward(self, _):
        self.dinputs = [[1 if element > 0 else 0 for element in sample] for sample in self.inputs] 

class Sigmoid(Activation):
    def __init__(self): # So that a new instace is not created rach time the forward() is run
        self.positive = lambda x: 1 / (exp(-x) + 1)
        self.negative = lambda x: exp(x) / (exp(x) + 1)
        self.__ID = "Sigmoid"

    def getID(self):
        return self.__ID

    def forward(self, inputs): # Squashes data between 0 and 1

        self.outputs = [self.negative(val[0]) if val[0] < 0 else self.positive(val[0]) for val in inputs] # avoids overflow errors with exp()

    def backward(self, dvalues):
        #print(dvalues)
        #input(self.outputs)
        self.dinputs = [[a*b*(1-b)] for a,b in zip(dvalues, self.outputs)]

# Loss
class BinaryCrossEntropy:  # Measure how well the model is.
    def forward(self, predictions, TrueVals):
        # Remove any 0s or 1s to avoid arithmethic errors
        predictions = clipEdges(predictions) # to 1e-16

        # Formula used: -(true * log(Predicted) + (1 - true) * log(1 - Predicted))
        SampleLoss = [-((tVal * log(pVal)) + ((1 - tVal) * log(1 - pVal))) for tVal, pVal in zip(TrueVals, predictions)]

        self.SampleLoss = sum(SampleLoss) / len(SampleLoss) # Average of all samples

    def backward(self, dvalues, TrueVals): # Dirative of above Formula

        dvalues = clipEdges(dvalues)

        # Formula: -(true / dvalues) + ( (1-true) / (1-dvalues))
        self.dinputs = [-(Tval/Dval) + (1-Tval)/(1-Dval) for Tval, Dval in zip(TrueVals, dvalues)]


class OptimizerSGD: # Broken
    def __init__(self, InitialLearningRate=1e-3, decay=1e-4, minimumLearningRate=1e-5, momentum=0.8): # learning rate too high = no learning
        self.InitialLearningRate = InitialLearningRate
        self.minimumLearningRate = minimumLearningRate
        self.decay = decay
        self.momentum = momentum

        self.activeLearningRate = InitialLearningRate

    def adjustLearningRate(self, iter):
        # Linear 
        self.activeLearningRate = max(self.InitialLearningRate / (1 + self.decay * iter), self.minimumLearningRate)

        # Exponentail:  INitail * e^(-(decay)(iter))
        # self.activeLearningRate = max(self.InitialLearningRate * exp(-(self.decay * iter)), self.minimumLearningRate)

    def UpdateParameters(self, layer):
        # Adjust Weights    |    layer.weightsVelocity = self.momentum * layer.weightsVelocity - self.activeLearningRate * layer.dweights

        # self.activeLearningRate * layer.dweights
        AdjustedDWeight = [DM.Multiply(self.activeLearningRate, sample) for sample in layer.dweights]
        # self.activeLearningRate * layer.dbiases
        AdjustedDBiases = DM.Multiply(self.activeLearningRate, layer.dbiases)

        if self.momentum != 0:
            # Adjust Weights    |    layer.weightsVelocity = self.momentum * layer.weightsVelocity - self.activeLearningRate * layer.dweights

            # self.momentum * layer.weightsVelocity
            newVelocity = [DM.Multiply(self.momentum, row) for row in layer.weightsVelocity]
            
            layer.weightsVelocity = [[a-b for a,b in zip(velocityRow, dweightsRow)] 
                                    for velocityRow, dweightsRow in zip(newVelocity, AdjustedDWeight)]
            

            # Adjust Biases    |    layer.biasesVelocity = self.momentum * layer.biasesVelocity - self.activeLearningRate * layer.dbiases

            # self.momentum * layer.biasesVelocity
            layer.biasesVelocity = [a-b for a,b in zip(DM.Multiply(self.momentum, layer.biases), AdjustedDBiases)]

            # Final updates
            layer.weights = [[a-b for a, b in zip(layer.weights[x], layer.weightsVelocity[x])] for x in range(len(layer.weights))]
            layer.biases = [a-b for a, b in zip(layer.biases, layer.biasesVelocity)]
        else:
            layer.weights = [[a-b for a, b in zip(layer.weights[x], AdjustedDWeight[x])] for x in range(len(layer.weights))]
            layer.biases = [a-b for a, b in zip(layer.biases, AdjustedDBiases)]
        

def clipEdges(list, scale=1e-7):
    for index, val in enumerate(list):
        if val < scale:
            list[index] = scale
        elif val > 1 - (scale):
            list[index] = 1 - (scale)
    return list