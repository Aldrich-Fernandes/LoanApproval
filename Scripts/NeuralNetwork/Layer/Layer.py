from .Activations import Sigmoid, ReLU

from Scripts import DataMethod as DM

from math import sqrt
from random import gauss

'''
Layer

Houses a collection of Neurons

This class automatically applies activation to the layer's output.

'''
class Layer:
    def __init__(self, NoOfInputs, NoOfNeurons, activation, regularisationStrength):
        self._NoOfInputs = NoOfInputs           # Number of neurons/inputs in the previous layer
        self._NoOfNeurons = NoOfNeurons         # Number of neurons/inputs in this layer

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

    # Creates new random weights and biases for the layer.
    def initialiseNewLayer(self): 
        # Xavier/Glorot weight initialization
        # Creates a dense layer initialised with a small value
        scale = sqrt(self.__Numerator / (self._NoOfInputs+self._NoOfNeurons))
        self.__weights = [[gauss(0, scale) for _ in range(self._NoOfNeurons)] 
                          for _ in range(self._NoOfInputs)] 
        self.__biases = [0.0 for _ in range(self._NoOfNeurons)]

        # Velocity for use by Optimiser
        self.__weightsVelocity = [[1e-3 for _ in range(self._NoOfNeurons)] for _ in range(self._NoOfInputs)]
        self.__biasesVelocity = [1e-3 for _ in range(self._NoOfNeurons)]

    # Formula = DotProduct(input, weights) + bias
    def forward(self, inputs):
        self.inputs = inputs.copy()

        self.output = [[a+b for a,b in zip(sample, self.__biases)] 
                       for sample in DM.DotProduct(inputs, self.__weights)]        

        self.activation.forward(self.output)

    # Calculates dvalues, dinputs, dweights, dbiases which are gradients that 
    # helps shows how much these attributes impacted the prediction
    def backward(self, dvalues):

        self.activation.backward(dvalues)
        dvalues = self.activation.dinputs.copy() # gradients from the activation function

        # Layer's dvalues (gradients) to be passes to the next layer
        self.dinputs = DM.DotProduct(dvalues, DM.Transpose(self.__weights)) 

        # Used by Optimiser to adjust weights and biases
        self.dweights = DM.DotProduct(DM.Transpose(self.inputs), dvalues)
        self.dbiases = [sum(x) for x in DM.Transpose(dvalues)]

        # L2 regularisation prevents a single weight from getting to large.
        if self.__regStr != 0:
            DweightsSqr = DM.Multiply(self.__regStr, self.__weights)
            self.dweights = [[a+(2*b) for a, b in zip(self.dweights[x], DweightsSqr[x])] 
                             for x in range(len(self.dweights))]

    # Getters and setters used by the optimiser to retrive and adjust the weights and biases
    def getVelocities(self):
        return self.__weightsVelocity, self.__biasesVelocity
    
    def setVelocities(self, veloWeights, veloBiases):
        self.__weightsVelocity, self.__biasesVelocity = veloWeights, veloBiases

    def getWeightsAndBiases(self):
        return self.__weights, self.__biases
    
    # Also used to load a pretrained model
    def setWeightsAndBiases(self, weights, biases):
        self.__weights, self.__biases = weights, biases