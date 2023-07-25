import numpy as np
from math import exp

from DataHandle import *

class NeuralNetwork:
    def __init__(self):
        self.TrainX, self.TrainY, self.TestX, self.TestY = PreProcess(100).getData()
        self.train()
        
    def train(self):
        Hiddenlayer = Layer(11, 7, ReLU())
        Outputlayer = Layer(7, 1, Sigmoid())

        Hiddenlayer.forward(self.TrainX)
        Outputlayer.forward(Hiddenlayer.output)
        
        self.result = Outputlayer.output

        self.CompareResults()

    def CompareResults(self):
        for i in range(20):
            x = random.randint(0,79)
            print(f"Predicted: {round(self.result[x])} Actual: {self.TrainY[x]}")
        
class Layer:
    def __init__(self, NoOfInputs, NoOfNeurons, activation):
        self.weights = 0.01 * np.random.randn(NoOfInputs, NoOfNeurons)
        self.biases = [0 for x in range(NoOfNeurons)]
        self.activation = activation

        self.output = []

    def forward(self, inputs):
        for entry in inputs:
            self.output.append([DataMethod.DotProduct(entry, WeightsForNeuron) + self.biases[NeuronIndex] for NeuronIndex, WeightsForNeuron in enumerate(DataMethod.Transpose(self.weights))])  # (1x11) dot (1x11)

        self.applyActivation()

    def applyActivation(self):
        self.output = self.activation.forward(self.output)

#Activations
class ReLU:
    def forward(self, inputs):
        for rowIndex, entry in enumerate(inputs):
            for index, element in enumerate(entry):
                if element < 0:
                    inputs[rowIndex][index] = 0
        return inputs

class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilites = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        return probabilites

class Sigmoid:
    def forward(self, inputs):

        return [round((1 / (1 + exp(-Val[0]))), 8) for Val in inputs]
        
