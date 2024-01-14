from ActivationLossAndOptimisers import ReLU, Sigmoid
from DataHandle import DataMethod as DM
from math import sqrt
from random import gauss

class Layer:
    def __init__(self, NoOfInputs, NoOfNeurons, activation="Sigmoid", regularisationStrength=0.001):
        self._NoOfInputs = NoOfInputs
        self._NoOfNeurons = NoOfNeurons

        # L2 regularisation - Adds a penalty to prevent overfitting and improve generalisation
        self.__regStr = regularisationStrength

        # Initilise Activation
        if activation == "Sigmoid":
            self.activation = Sigmoid()
            self.__Numerator = 1
        elif activation == "ReLU":
            self.activation = ReLU()
            self.__Numerator = 2

        self.initialiseNewLayer()

    def initialiseNewLayer(self): # Overloading
        # Xavier/Glorot weight initialization
        # Creates a dense layer where each neuron is connected to all the prior layer neurons with a small weight
        scale = sqrt(self.__Numerator / (self._NoOfInputs+self._NoOfNeurons))
        self.__weights = [[gauss(0, scale) for _ in range(self._NoOfNeurons)] for _ in range(self._NoOfInputs)] 
        self.__biases = [0.0 for x in range(self._NoOfNeurons)]

        # Velocity for use with Optimizer momentum - Allows the model to move in one direction (reduces accurcay fluctuations)
        self.__weightsVelocity = [[1e-3 for _ in range(self._NoOfNeurons)] for _ in range(self._NoOfInputs)]
        self.__biasesVelocity = [1e-3 for _ in range(self._NoOfNeurons)]

    # Formula = sum(weights x input) + bias
    def forward(self, inputs):
        self.inputs = inputs.copy()

        self.output = [[a+b for a,b in zip(sample, self.__biases)] for sample in DM.DotProduct(inputs, self.__weights)]        

        self.activation.forward(self.output)

    def backward(self, dvalues):
        # dvalues, dinputs, dweights, dbiases: are gradients show how much it impacted the results

        self.activation.backward(dvalues)
        dvalues = self.activation.dinputs.copy()

        self.dinputs = DM.DotProduct(dvalues, DM.Transpose(self.__weights)) # Layer's dvalues

        self.dweights = DM.DotProduct(DM.Transpose(self.inputs), dvalues)   # used by optimizer to adjust weights

        self.dbiases = [sum(x) for x in DM.Transpose(dvalues)]              # used by optimizer to adjist biases

        # L2 regularisation prevents a single weight from getting to large.
        if self.__regStr != 0:
            DweightsRegularisation = DM.Multiply(self.__regStr, self.__weights)
            self.dweights = [[a+(2*b) for a, b in zip(self.dweights[x], DweightsRegularisation[x])] 
                             for x in range(len(self.dweights))]

    # Getters and Setters
    def getVelocities(self):
        return self.__weightsVelocity, self.__biasesVelocity
    
    def setVelocities(self, veloWeights, veloBiases):
        self.__weightsVelocity, self.__biasesVelocity = veloWeights, veloBiases

    def getWeightsAndBiases(self):
        return self.__weights, self.__biases
    
    def setWeightsAndBiases(self, weights, biases):
        self.__weights, self.__biases = weights, biases