import numpy as np
from math import exp, log

from DataHandle import *

DM = DataMethod()

class NeuralNetwork:
    def __init__(self):
        self.TrainX, self.TrainY, self.TestX, self.TestY = PreProcess(100).getData()
        self.Loss = 0.0
        self.Accuracy = 0.0
        self.train()
        
    def train(self):
        #Create Network
        Hiddenlayer = Layer(11, 7, ReLU())
        Outputlayer = Layer(7, 1, Sigmoid())

        Hiddenlayer.forward(self.TrainX)
        Outputlayer.forward(Hiddenlayer.output)
        
        self.result = Outputlayer.output

        # Initail Loss         
        BinaryLoss = BinaryCrossEntropy()
        self.Loss = BinaryLoss.calculate(self.result, self.TrainY)

        self.UpdateAccuracy()
        self.CompareResults()

    def CompareResults(self):
        for i in range(10):
            x = random.randint(0,79)
            print(f"Predicted: {round(self.result[x], 8)} Actual: {self.TrainY[x]}")
        print(f"Loss: {self.Loss} \nAccuracy: {self.Accuracy}")

    def UpdateAccuracy(self):
        self.Accuracy = sum([1 for x,y in zip(self.result, self.TrainY) if round(x)==y]) / len(self.result)
        
class Layer:
    def __init__(self, NoOfInputs, NoOfNeurons, activation):
        self.weights = 0.01 * np.random.randn(NoOfInputs, NoOfNeurons)
        self.biases = [0 for x in range(NoOfNeurons)]
        self.activation = activation

        self.output = []

    def forward(self, inputs):
        
        self.output = [[DM.DotProduct(entry, WeightsForNeuron) + self.biases[NeuronIndex] for NeuronIndex, WeightsForNeuron in enumerate(DM.Transpose(self.weights))] 
                       for entry in inputs]

        #self.output = [[round(sum([x*y for x,y in zip(entry, WeightsForNeuron)]), 8) + self.biases[NeuronIndex] for NeuronIndex, WeightsForNeuron in enumerate([[self.weights[x][y] for x in range(len(self.weights))] for y in range(len(self.weights[0]))])] for entry in inputs]
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
        
# Loss
class Loss:
    def calculate(self, output, y):
        SampleLosses = self.forward(output, y)

        DataLoss = np.mean(SampleLosses)

        return DataLoss

class BinaryCrossEntropy(Loss):
    def forward(self, predictions, TrueVals):
        # Remove any 0s or 1s to avoid arithmethic errors
        for index, val in enumerate(predictions):
            if val < 1e-7:
                predictions[index] = 0.0000001
            elif val > 1- 1e-7:
                predictions[index] = 0.9999999


        SampleLoss = [-(val1+val2) for val1, val2 in zip(DM.Multiply(TrueVals, [log(x) for x in predictions]), 
                                                        DM.Multiply([1-x for x in TrueVals], [log(1-x) for x in predictions]))]
        SampleLoss = np.mean(SampleLoss, axis=-1)

        return SampleLoss
