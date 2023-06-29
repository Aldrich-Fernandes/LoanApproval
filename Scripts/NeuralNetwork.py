from DataHandle import *
import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.currentDataset = 0
        self.inputs = [[4, 6 , 3, 3],       # Dataset 1
                       [3, 56, 3, 2],       # Dataset 2
                       [2.4, 34, 5.2, 14]]  # Dataset 3
        
        self.weights = [[1, 3, 5, 3],       # Weights for first neuron in next layer
                        [43, 5.3, 13, 35],  # Weights for second neuron in next layer
                        [42, 3.2, 2, 4.3]]  # Weights for third neuron in next layer

        self.bias = [2.2, 4, 5] # For each next neuron

        self.output = np.dot(self.inputs, self.weights) + self.bias
        self.hiddenLayer = NeuralLayer()
        self.outputLayer = NeuralLayer()


    def NextDataset(self):
        self.currentDataset += 1


class NeuralLayer:
    def __init__(self, NoOfInputs, NoOfNeurons):
        self.weights = 0.01 
        self.bias = []

    def ForwardPropagation(self, inputs):
        # dot product
        pass

    def BackPropagation(self):
        raise NotImplementedError
