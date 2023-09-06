import numpy as np

from DataHandle import *
from ActivationLossAndOptimizers import ReLU, Sigmoid, BinaryCrossEntropy

DM = DataMethod()

class NeuralNetwork:        
    def train(self, mode, NumOfDatasets=300):
        # Important Values
        self.TrainX, self.TrainY, self.TestX, self.TestY = PreProcess(mode, NumOfDatasets).getData() # max 614
        self.Loss = 0.0
        self.Accuracy = 0.0

        #Create Network
        Hiddenlayer1 = Layer(11, 7, ReLU())
        Hiddenlayer2 = Layer(7, 4, ReLU())
        Outputlayer = Layer(4, 1, Sigmoid())

        BinaryLoss = BinaryCrossEntropy()

        # Training Values
        LowestLoss = 9999999
        Epochs = 2000

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
            Hiddenlayer2.forward(Hiddenlayer1.activation.outputs)
            Outputlayer.forward(Hiddenlayer2.activation.outputs)

            result = Outputlayer.activation.outputs.copy()

            self.Loss = BinaryLoss.calculate(result, self.TrainY)

            self.Accuracy = sum([1 for x,y in zip(result, self.TrainY) if round(x)==y]) / len(result)
            
            if self.Loss < LowestLoss:
                self.DisplayResults(iteration, LowestLoss)

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

            if iteration % 100 == 0:
                self.DisplayResults(iteration, LowestLoss) 

            BinaryLoss.backward(result, self.TrainY)
            Outputlayer.backward(BinaryLoss.dinputs)
            Hiddenlayer2.backward(Outputlayer.dinputs)
            Hiddenlayer1.backward(Hiddenlayer2.dinputs)
            
        # test
        Hiddenlayer1.forward(self.TestX)
        Hiddenlayer2.forward(Hiddenlayer1.activation.outputs)
        Outputlayer.forward(Hiddenlayer2.activation.outputs)

        result = Outputlayer.activation.outputs.copy()
        for x in result:
            print(x)
        for x in range(len(result)):
            print(f"True: {self.TestY[x]} Predicted: {round(result[x])}")
        
        print(sum([1 for x,y in zip(result, self.TestY) if round(x)==y]) / len(result))

    def DisplayResults(self, iteration, LowestLoss):
        print(f"Iteration: {iteration} Loss: {round(LowestLoss, 5)} Accuracy: {round(self.Accuracy, 5)}\n\n")

    
        
class Layer:
    def __init__(self, NoOfInputs, NoOfNeurons, activation):
        self.__NoOfInputs = NoOfInputs
        self.__NoOfNeurons = NoOfNeurons
        self.weights = [DM.Multiply(0.01, np.random.randn(1, NoOfNeurons).tolist()[0])
                       for i in range(NoOfInputs)]
    
        self.biases = [0.0 for x in range(NoOfNeurons)]
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs.copy()

        self.output = [[a+b for a,b in zip(sample, self.biases)] for sample in DM.DotProduct(inputs, self.weights)] # add biases  -- (10x7)/ (samplesize x NOofNeurons)        

        self.activation.forward(self.output)

    def backward(self, dvalues):
        self.activation.backward(dvalues)

        dvalues = self.activation.dinputs.copy()

        self.dweights = [DM.DotProduct(DM.Transpose(self.inputs), dvalues)]
        self.dbiases = sum([x[0] for x in dvalues])

        self.dinputs = DM.DotProduct(dvalues, DM.Transpose(self.weights))

    def incrementVals(self, multiplier=0.05):
        FractionIncrease = [DM.Multiply(multiplier, np.random.randn(1, self.__NoOfNeurons).tolist()[0])
                       for sample in range(self.__NoOfInputs)]
        
        self.weights = [[a+b for a,b in zip(FractionIncrease[i], self.weights[i])] for i in range(self.__NoOfInputs)]

        self.biases = [a+b for a,b in zip(self.biases, DM.Multiply(multiplier, 
                                                                   np.random.randn(1, self.__NoOfNeurons).tolist()[0]))]
        
    