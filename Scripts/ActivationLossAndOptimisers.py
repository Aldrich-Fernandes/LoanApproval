from math import exp, log
from abc import ABC, abstractmethod

from DataHandle import DataMethod as DM

'''
Activations

Introduces more flexibility to the network, allowing it to understand non-linear and more complex 
relationships and patterns in the data/between features.

'''
class Activation(ABC):
    # Contains formula to pass data through
    @abstractmethod
    def forward(self, inputs): 
        pass

    # Uses the derivative to calculate Dvalues (gradients) which are used to minimise the loss.
    @abstractmethod
    def backward(self, dvalues):
        pass

class ReLU(Activation): # Rectified Linear Unit
    # limits inputs between (>= 0)
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = [[max(0, element) for element in entry]
                        for entry in inputs]

    def backward(self, _):
        self.dinputs = [[1 if element > 0 else 0 for element in sample] for sample in self.inputs] 

class Sigmoid(Activation):
    def __init__(self): 
        # So that a new instance is not created each time the forward() is run
        self.__positive = lambda x: 1 / (exp(-x) + 1)
        self.__negative = lambda x: exp(x) / (exp(x) + 1)

    # Squashes data between 0 and 1
    def forward(self, inputs): 

        # Prevents overflow errors with the exp() function
        self.outputs = [self.__negative(val[0]) if val[0] < 0 else self.__positive(val[0]) for val in inputs]

    def backward(self, dvalues):
        self.dinputs = [[a*b*(1-b)] for a,b in zip(dvalues, self.outputs)]

'''
Loss 

Measures how well the model performed by comparing the true and predicted values

The algorithm doesn't utilise the calculated loss value directly. It is use to visualise if the model is
improving when training and identifying what is impacting the model and how much does it. 
'''
class BinaryCrossEntropy:
    def __init__(self, regStr=0.0):
        self.__sampleLoss = 0.0
        self.__regLoss = 0.0

        # How strongly to penalise the model  
        self.__regStr = regStr

    def forward(self, predictions, TrueVals):
        predictions = clipEdges(predictions)

        # Formula used: -(true * log(Predicted) + (1 - true) * log(1 - Predicted))
        sampleLoss = [-((tVal * log(pVal)) + ((1 - tVal) * log(1 - pVal))) 
                      for tVal, pVal in zip(TrueVals, predictions)]

        self.__sampleLoss = sum(sampleLoss) / len(sampleLoss)       # Average of all samples

        self.__regLoss = 0                                          # Resets the loss for that epoch

    def backward(self, predicted, TrueVals): 
        predicted = clipEdges(predicted)
        
        # Derivative of formula above used: (PredictVal - Tval) / ((1-PredictVal) * PredictVal)
        self.dinputs = [(PredictVal - Tval) / ((1-PredictVal) * PredictVal) 
                        for Tval, PredictVal in zip(TrueVals, predicted)]

    # For adjust the hyperparameter when training a new model
    def updateRegStr(self, regStr):
        self.__regStr = regStr

    # L2 regularisation foumula: 0.5 * regStr * SumOfSquaredWeights
    def calcRegularisationLoss(self, layerWeights):
        if self.__regStr != 0:
            weightSqrSum = sum([sum(x) for x in DM.Multiply(layerWeights, layerWeights)])

            self.__regLoss += 0.5 * self.__regStr * weightSqrSum

    def getLoss(self):
        return self.__sampleLoss + self.__regLoss

'''
Optimser

Improve the accuracy of the model.

Does this by adjusting the weights and biases of layers by adding/subtracting a small amount to the weights 
depending on their impact on the model and its output which is calculated in the backpass (utilises the 
dvalues).
'''
class OptimiserSGD:
    def __init__(self, InitialLearningRate=1e-4, decay=5e-5, momentum=0.95, mode="Linear"):
        self.__InitialLearningRate = InitialLearningRate          # Starting Learning rate
        self.__minimumLearningRate = InitialLearningRate * 0.001  # Lower bound Learning rate
        self.__decay = decay                                      # Rate at which Learning rate decreases
        self.__momentum = momentum                                # Promotes adjustment movement in one direction
        self.activeLearningRate = InitialLearningRate             # How much to adjust/step. 

        self.__mode = mode

    # Gradually decreases the learning rate to avoid overshooting the optimal parameters
    # If it is too high it will overshoot the optimal but if too low the mode won't train properly.
    def adjustLearningRate(self, iter): 
        if self.__decay != 0:
            if self.__mode == "Linear":
                newLearningRate = self.__InitialLearningRate / (1 + self.__decay * iter)
            elif self.__mode == "Exponential":
                newLearningRate = self.__InitialLearningRate * exp(-self.__decay * iter)

            self.activeLearningRate = max(newLearningRate, self.__minimumLearningRate)

    # Function to update the parameters of a neural network layer using SGD with momentum
    def UpdateParameters(self, layer): 

        # Amount to increment the weights and biases
        weightUpdate = DM.Multiply(self.activeLearningRate, layer.dweights)
        biasesUpdate = DM.Multiply(self.activeLearningRate, layer.dbiases)

        weights, biases = layer.getWeightsAndBiases()

        if self.__momentum != 0:

            # Amount to add to the weights and biases to reduce fluctuations in accuracy and loss
            weightVelocityUpdate, biasesVelocityUpdate = layer.getVelocities()

            # New weight velocity = momentum * Velocity - activeLearningRate * dweights
            weightVelocityUpdate = [[a - b for a, b in zip(velocityRow, dweightsRow)]
                               for velocityRow, dweightsRow in zip(DM.Multiply(self.__momentum, weightVelocityUpdate),
                                                                  weightUpdate)]

            # New bias velocity = momentum * Velocity - activeLearningRate * dbiases
            biasesVelocityUpdate = [a - b for a, b in zip(DM.Multiply(self.__momentum, biasesVelocityUpdate), 
                                                          biasesUpdate)]

            layer.setVelocities(weightVelocityUpdate, biasesVelocityUpdate)

            # Final Updates
            weights = [[a + b for a, b in zip(weights[x], weightVelocityUpdate[x])] for x in range(len(weights))]
            biases = [a + b for a, b in zip(biases, biasesVelocityUpdate)]
        else:
            weights = [[a - b for a, b in zip(weights[x], weightUpdate[x])] for x in range(len(weights))]
            biases = [a - b for a, b in zip(biases, biasesUpdate)]

        layer.setWeightsAndBiases(weights, biases)

# Replaces any 0s or 1s to avoid arithmetic errors
def clipEdges(list, scale=1e-7):
    for index, val in enumerate(list):
        if val < scale:
            list[index] = scale
        elif val > 1 - (scale):
            list[index] = 1 - (scale)
    return list