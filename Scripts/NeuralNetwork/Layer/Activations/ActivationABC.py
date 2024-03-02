from abc import ABC, abstractmethod

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