from math import exp, log
from abc import ABC, abstractmethod

from DataHandle import DataMethod as DM

#Activations
class Activation(ABC): # Scales the output of a layer
    @abstractmethod
    def forward(self, inputs): # Formula
        pass

    @abstractmethod
    def backward(self, dvalues): # Diretive of the Formula
        pass

class ReLU(Activation): # Rectified Linear Unit
    # limits between (>= 0)
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = [[max(0, element) for element in entry]
                        for entry in inputs]

    def backward(self, _):
        self.dinputs = [[1 if element > 0 else 0 for element in sample] for sample in self.inputs] 

class Sigmoid(Activation):
    def __init__(self): 
        # So that a new instace is not created each time the forward() is run
        self.positive = lambda x: 1 / (exp(-x) + 1)
        self.negative = lambda x: exp(x) / (exp(x) + 1)

    # Squashes data between 0 and 1
    def forward(self, inputs): 

        # Introduced different 
        self.outputs = [self.negative(val[0]) if val[0] < 0 else self.positive(val[0]) for val in inputs] # avoids overflow errors with exp()

    def backward(self, dvalues):
        self.dinputs = [[a*b*(1-b)] for a,b in zip(dvalues, self.outputs)]

# Loss - Measure how well the model performed
class BinaryCrossEntropy:
    def __init__(self, regularisationStrenght=0):
        self.sampleLoss = 0
        self.regularisationLoss = 0
        self.regularisationStrenght = regularisationStrenght

    def forward(self, predictions, TrueVals):
        # Remove any 0s or 1s to avoid arithmethic errors
        predictions = clipEdges(predictions)

        # Formula used: -(true * log(Predicted) + (1 - true) * log(1 - Predicted))
        sampleLoss = [-((tVal * log(pVal)) + ((1 - tVal) * log(1 - pVal))) for tVal, pVal in zip(TrueVals, predictions)]

        self.sampleLoss = sum(sampleLoss) / len(sampleLoss) # Average of all samples

        self.regularisationLoss = 0 # Reset for that epoch

    # Dirative of above Formula
    def backward(self, predicted, TrueVals): 
        predicted = clipEdges(predicted)
        
        # Formula == (PredictVal - Tval) / ((1-PredictVal) * PredictVal)
        self.dinputs = [(PredictVal - Tval) / ((1-PredictVal) * PredictVal) for Tval, PredictVal in zip(TrueVals, predicted)]

    def calcregularisationLoss(self, layerWeights):
        
        if self.regularisationStrenght != 0:
            self.regularisationLoss += 0.5 * self.regularisationStrenght * sum([sum(x) for x in DM.Multiply(layerWeights, layerWeights)])

    def getLoss(self):
        return self.sampleLoss + self.regularisationLoss


class OptimizerSGD:
    def __init__(self, InitialLearningRate=0.01, decay=1e-4, minimumLearningRate=1e-5, momentum=0.9):
        self.InitialLearningRate = InitialLearningRate          # Starting Learning rate
        self.minimumLearningRate = minimumLearningRate          # Lower bound Leanring rate
        self.decay = decay                                      # Rate at which Learning rate decreases
        self.momentum = momentum                                # Makes Accuracy and Loss change in a consistant way in one direction
        self.activeLearningRate = InitialLearningRate           # Working learning rate

    def adjustLearningRate(self, iter, mode="Linear"): # gradually decreases the learning rate to avoid overshooting the optimal parameters
        if self.decay != 0:
            if mode == "Linear":
                self.activeLearningRate = max(self.InitialLearningRate / (1 + self.decay * iter), self.minimumLearningRate)
            elif mode == "Exponential":
                self.activeLearningRate = max(self.InitialLearningRate * exp(-self.decay * iter), self.minimumLearningRate)

    # Function to update the parameters of a neural network layer using SGD with momentum
    def UpdateParameters(self, layer): 
        AdjustedDWeight = DM.Multiply(self.activeLearningRate, layer.dweights)
        AdjustedDBiases = DM.Multiply(self.activeLearningRate, layer.dbiases)

        weights, biases = layer.getWeightsAndBiases()

        if self.momentum != 0:
            weightsVelocity, biasesVelocity = layer.getVelocities()

            # Adjust Weights    |    layer.weightsVelocity = self.momentum * layer.weightsVelocity - self.activeLearningRate * layer.dweights

            weightsVelocity = [[a - b for a, b in zip(velocityRow, dweightsRow)]
                               for velocityRow, dweightsRow in zip(DM.Multiply(self.momentum, weightsVelocity),
                                                                  AdjustedDWeight)]

            # Adjust Biases    |    layer.biasesVelocity = self.momentum * layer.biasesVelocity - self.activeLearningRate * layer.dbiases
            biasesVelocity = [a - b for a, b in zip(DM.Multiply(self.momentum, biasesVelocity), AdjustedDBiases)]

            layer.setVelocities(weightsVelocity, biasesVelocity)

            # Final Updates

            weights = [[a + b for a, b in zip(weights[x], weightsVelocity[x])] for x in range(len(weights))]
            biases = [a + b for a, b in zip(biases, biasesVelocity)]
        else:
            weights = [[a - b for a, b in zip(weights[x], AdjustedDWeight[x])] for x in range(len(weights))]
            biases = [a - b for a, b in zip(biases, AdjustedDBiases)]

        layer.setWeightsAndBiases(weights, biases)

def clipEdges(list, scale=1e-7):
    for index, val in enumerate(list):
        if val < scale:
            list[index] = scale
        elif val > 1 - (scale):
            list[index] = 1 - (scale)
    return list