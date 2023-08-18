import numpy as np
from math import exp, log

from DataHandle import *
from ActivationLossAndOptimizers import *

DM = DataMethod()

class NeuralNetwork:
    def __init__(self):
        self.train()
        
    def train(self):
        # Important Values
        self.TrainX, self.TrainY, self.TestX, self.TestY = PreProcess(NumOfDatasets=300).getData() # max 614
        self.Loss = 0.0
        self.Accuracy = 0.0

        #Create Network
        Hiddenlayer1 = Layer(11, 7, ReLU())
        Hiddenlayer2 = Layer(7, 4, ReLU())
        Outputlayer = Layer(4, 1, Sigmoid())

        BinaryLoss = BinaryCrossEntropy()

        # Training Values
        LowestLoss = 9999999
        Epochs = 10000

        BestWeight_H1 = Hiddenlayer1.weights.copy()
        BestBiases_H1 = Hiddenlayer1.biases.copy()

        BestWeight_H2 = Hiddenlayer2.weights.copy()
        BestBiases_H2 = Hiddenlayer2.biases.copy()

        BestWeight_O = Outputlayer.weights.copy()
        BestBiases_O = Outputlayer.biases.copy()

        # Epochs
        for iteration in range(Epochs):
        
            Hiddenlayer1.incrementVals()
            Hiddenlayer2.incrementVals()
            Outputlayer.incrementVals()

            Hiddenlayer1.forward(self.TrainX)
            Hiddenlayer2.forward(Hiddenlayer1.output)
            Outputlayer.forward(Hiddenlayer2.output)

            result = Outputlayer.output    
            
            self.Loss = BinaryLoss.calculate(Outputlayer.output, self.TrainY)
            self.Accuracy = sum([1 for x,y in zip(result, self.TrainY) if round(x)==y]) / len(result)
            
            if self.Loss < LowestLoss:
                self.DisplayResults(iteration)

                BestWeight_H1 = Hiddenlayer1.weights.copy()
                BestBiases_H1 = Hiddenlayer1.biases.copy()

                BestWeight_H2 = Hiddenlayer2.weights.copy()
                BestBiases_H2 = Hiddenlayer2.biases.copy()

                BestWeight_O = Outputlayer.weights.copy()
                BestBiases_O = Outputlayer.biases.copy()

                LowestLoss = self.Loss
            else:
                Hiddenlayer1.weights = BestWeight_H1.copy()
                Hiddenlayer1.biases = BestBiases_H1.copy()

                Hiddenlayer2.weights = BestWeight_H2.copy()
                Hiddenlayer2.biases = BestBiases_H2.copy()

                Outputlayer.weights = BestWeight_O.copy()
                Outputlayer.biases = BestBiases_O.copy()

    def DisplayResults(self, iteration):
        print(f"Iteration: {iteration} Loss: {round(self.Loss, 5)} Accuracy: {round(self.Accuracy, 5)}\n\n")

    def SaveModel(self): ## Saves Best Weights, Biases and categorical Feature key into a txt file
        pass

    def LoadModel(self): ## Loads Best Weights and Biases from txt file
        pass
        
class Layer:
    def __init__(self, NoOfInputs, NoOfNeurons, activation):
        self.NoOfInputs = NoOfInputs
        self.NoOfNeurons = NoOfNeurons
        self.weights = [DM.Multiply([0.01 for x in range(NoOfInputs)], np.random.randn(1, self.NoOfNeurons).tolist()[0])
                       for sample in range(self.NoOfInputs)]
    
        self.biases = [0.0 for x in range(NoOfNeurons)]
        self.activation = activation

        self.output = []

    def forward(self, inputs):
        
        self.output = [[DM.DotProduct(entry, WeightsForNeuron) + self.biases[NeuronIndex] for NeuronIndex, WeightsForNeuron in enumerate(DM.Transpose(self.weights))] 
                       for entry in inputs]

        #self.output = [[round(sum([x*y for x,y in zip(entry, WeightsForNeuron)]), 8) + self.biases[NeuronIndex] for NeuronIndex, WeightsForNeuron in enumerate([[self.weights[x][y] for x in range(len(self.weights))] for y in range(len(self.weights[0]))])] for entry in inputs]
        self.applyActivation()

    def applyActivation(self):
        self.output = self.activation.forward(self.output)

    def incrementVals(self, multiplier=0.05):
        self.weights += multiplier * np.random.randn(self.NoOfInputs, self.NoOfNeurons)
        self.biases = [a+b for a,b in zip(self.biases, DM.Multiply([multiplier for x in range(self.NoOfNeurons)], 
                                                                   np.random.randn(1, self.NoOfNeurons).tolist()[0]))]


