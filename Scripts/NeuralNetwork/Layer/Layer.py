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
        self.__NoOfInputs = NoOfInputs           # Number of neurons/inputs in the previous layer
        self.__NoOfNeurons = NoOfNeurons         # Number of neurons/inputs in this layer

        # L2 regularisation - Adds a penalty to prevent overfitting and improve generalisation
        self.__regStr = regularisationStrength

        # Initialize Activation
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
        scale = sqrt(self.__Numerator / (self.__NoOfInputs+self.__NoOfNeurons))
        self._weights = [[gauss(0, scale) for _ in range(self.__NoOfNeurons)] 
                          for _ in range(self.__NoOfInputs)] 
        self._biases = [0.0 for _ in range(self.__NoOfNeurons)]

        # Velocity for use by Optimiser
        self._weightsVelocity = [[1e-3 for _ in range(self.__NoOfNeurons)] for _ in range(self.__NoOfInputs)]
        self._biasesVelocity = [1e-3 for _ in range(self.__NoOfNeurons)]

    # Formula = DotProduct(input, weights) + bias
    def forward(self, inputs):
        self.inputs = inputs.copy()

        self.output = [[a+b for a,b in zip(sample, self._biases)] 
                       for sample in DM.DotProduct(inputs, self._weights)]        

        self.activation.forward(self.output)

    # Calculated dvalues, dinputs, dweights, dbiases which are gradients that 
    # helps shows how much these attributes impacted the prediction
    def backward(self, dvalues):

        self.activation.backward(dvalues)
        dvalues = self.activation.dinputs.copy() # gradients from the activation function

        # Layer's dvalues (gradients) to be passes to the next layer
        self.dinputs = DM.DotProduct(dvalues, DM.Transpose(self._weights)) 

        # Used by Optimiser to adjust weights and biases
        self.dweights = DM.DotProduct(DM.Transpose(self.inputs), dvalues)
        self.dbiases = [sum(x) for x in DM.Transpose(dvalues)]

        # L2 regularisation prevents a single weight from getting to large.
        if self.__regStr != 0:
            DweightsSqr = DM.Multiply(self.__regStr, self._weights)
            self.dweights = [[a+(2*b) for a, b in zip(self.dweights[x], DweightsSqr[x])] 
                             for x in range(len(self.dweights))]

    # Getters and setters used by the optimiser to retrieve and adjust the weights and biases
    def getVelocities(self):
        return self._weightsVelocity, self._biasesVelocity
    
    def setVelocities(self, veloWeights, veloBiases):
        self._weightsVelocity, self._biasesVelocity = veloWeights, veloBiases

    def getWeightsAndBiases(self):
        return self._weights, self._biases
    
    # Also used to load a pretrained model
    def setWeightsAndBiases(self, weights, biases):
        self._weights, self._biases = weights, biases