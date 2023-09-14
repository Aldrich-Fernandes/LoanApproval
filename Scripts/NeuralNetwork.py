import numpy as np

from DataHandle import *
from ActivationLossAndOptimizers import ReLU, Sigmoid, BinaryCrossEntropy, OptimizerSGD

DM = DataMethod()

class NeuralNetwork:        # Currently overfitting
    def train(self, mode, NumOfDatasets=300):
        # Important Values
        self.TrainX, self.TrainY, self.TestX, self.TestY = PreProcess(mode, NumOfDatasets).getData() # max 614
        self.Accuracy = 0.0

        #Create Network
        Hiddenlayer = Layer(11, 7, ReLU())
        Outputlayer = Layer(7, 1, Sigmoid())

        BinaryLoss = BinaryCrossEntropy()
        Optimizer = OptimizerSGD()

        # Training Values
        self.LowestLoss = 9999999
        Epochs = 40

        # Epochs
        for iteration in range(Epochs):
        
            Hiddenlayer.forward(self.TrainX)
            Outputlayer.forward(Hiddenlayer.activation.outputs)

            result = Outputlayer.activation.outputs.copy()
            loss = BinaryLoss.forward(result, self.TrainY)

            self.Accuracy = sum([1 for x,y in zip(result, self.TrainY) if round(x)==y]) / len(result)
            
            if loss < self.LowestLoss:
                self.LowestLoss = loss
            
            self.DisplayResults(iteration) 

            BinaryLoss.backward(result, self.TrainY)
            Outputlayer.backward(BinaryLoss.dinputs)
            Hiddenlayer.backward(Outputlayer.dinputs)

            Optimizer.UpdateParameters(Hiddenlayer)
            Optimizer.UpdateParameters(Outputlayer)
            
        # test
        Hiddenlayer.forward(self.TestX)
        Outputlayer.forward(Hiddenlayer.activation.outputs)

        result = Outputlayer.activation.outputs.copy()
        for x in result:
            print(x)
        for x in range(len(result)):
            print(f"True: {self.TestY[x]} Predicted: {round(result[x])}")
        
        print(sum([1 for x,y in zip(result, self.TestY) if round(x)==y]) / len(result))

    def DisplayResults(self, iteration):
        print(f"Iteration: {iteration} Loss: {round(self.LowestLoss, 5)} Accuracy: {round(self.Accuracy, 5)}\n\n")
        
class Layer:
    def __init__(self, NoOfInputs, NoOfNeurons, activation):
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

        self.dinputs = DM.DotProduct(dvalues, DM.Transpose(self.weights))

        self.dweights = DM.DotProduct(DM.Transpose(self.inputs), dvalues)
        self.dbiases = [sum(x) for x in dvalues]