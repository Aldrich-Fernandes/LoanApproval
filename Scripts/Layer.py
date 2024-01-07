from ActivationLossAndOptimisers import ReLU, Sigmoid
from DataHandle import DataMethod as DM
from math import sqrt
from random import gauss

class Layer:
    def __init__(self, NoOfInputs, NoOfNeurons, activation="Sigmoid", regularisationStrenght=0.001):
        self._NoOfInputs = NoOfInputs
        self._NoOfNeurons = NoOfNeurons

        # L2 regularisation - Adds a penalty to prevent overfitting and improve generalisation
        self.__regularisationStrenght = regularisationStrenght

        # Initilise Activation
        if activation == "Sigmoid":
            self.activation = Sigmoid()
            self.__Numerator = 1
        elif activation == "ReLU":
            self.activation = ReLU()
            self.__Numerator = 2

        self.SetWeightsAndBiases()

    def SetWeightsAndBiases(self):
        # Xavier/Glorot weight initialization
        scale = sqrt(self.__Numerator / (self._NoOfInputs+self._NoOfNeurons))
        self.__weights = [[gauss(0, scale) for _ in range(self._NoOfNeurons)] for _ in range(self._NoOfInputs)]
        self.__biases = [0.0 for x in range(self._NoOfNeurons)]

        # Velocity for use with Optimizer momentum
        self.__weightsVelocity = [[1e-3 for _ in range(self._NoOfNeurons)] for _ in range(self._NoOfInputs)]
        self.__biasesVelocity = [1e-3 for _ in range(self._NoOfNeurons)]

    # Formula = sum(weights x input) + bias
    def forward(self, inputs):
        self.inputs = inputs.copy()

        self.output = [[a+b for a,b in zip(sample, self.__biases)] for sample in DM.DotProduct(inputs, self.__weights)]        

        self.activation.forward(self.output)

    def backward(self, dvalues):
        self.activation.backward(dvalues)
        dvalues = self.activation.dinputs.copy()

        self.dinputs = DM.DotProduct(dvalues, DM.Transpose(self.__weights)) # Layer's dvalues

        self.dweights = DM.DotProduct(DM.Transpose(self.inputs), dvalues)   # used by optimizer to adjust weights

        self.dbiases = [sum(x) for x in DM.Transpose(dvalues)]              # used by optimizer to adjist biases

        if self.__regularisationStrenght != 0:                              # adds penalty
            DweightsRegularisation = DM.Multiply(self.__regularisationStrenght, self.__weights)
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