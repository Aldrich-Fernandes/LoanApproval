from .ActivationABC import Activation
from math import exp

class Sigmoid(Activation):
    def __init__(self): 
        # So that a new instance is not created each time the forward() is run
        # Prevents overflow errors with the exp() function
        self.__positive = lambda x: 1 / (exp(-x) + 1)
        self.__negative = lambda x: exp(x) / (exp(x) + 1)

    # Squashes data between 0 and 1
    def forward(self, inputs): 
        self.outputs = [self.__negative(val[0]) if val[0] < 0 else self.__positive(val[0]) for val in inputs]

    def backward(self, dvalues):
        self.dinputs = [[a*b*(1-b)] for a,b in zip(dvalues, self.outputs)]