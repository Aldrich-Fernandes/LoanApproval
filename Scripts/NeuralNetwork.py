import numpy as np

from DataHandle import *
from ActivationLossAndOptimizers import ReLU, Sigmoid, BinaryCrossEntropy, OptimizerSGD

DM = DataMethod()

class NeuralNetwork:        
    def train(self, mode, NumOfDatasets=300):
        # Important Values
        self.TrainX, self.TrainY, self.TestX, self.TestY = PreProcess(mode, NumOfDatasets).getData() # max 614
        self.Loss = 0.0
        self.Accuracy = 0.0

        #Create Network
        Hiddenlayer1 = Layer(11, 7, ReLU())
        #Hiddenlayer2 = Layer(7, 4, ReLU())
        Outputlayer = Layer(7, 1, Sigmoid())

        BinaryLoss = BinaryCrossEntropy()
        Optimizer = OptimizerSGD()

        # Training Values
        LowestLoss = 9999999
        Epochs = 2000    

        # Epochs
        for iteration in range(Epochs):
        
            Hiddenlayer1.incrementVals()
            #Hiddenlayer2.incrementVals()
            Outputlayer.incrementVals()

            Hiddenlayer1.forward(self.TrainX)
            #Hiddenlayer2.forward(Hiddenlayer1.activation.outputs)
            Outputlayer.forward(Hiddenlayer1.activation.outputs)

            result = Outputlayer.activation.outputs.copy()
            #input(result)
            self.Loss = BinaryLoss.calculate(result, self.TrainY)

            self.Accuracy = sum([1 for x,y in zip(result, self.TrainY) if round(x)==y]) / len(result)
            
            if self.Loss < LowestLoss:
                LowestLoss = self.Loss
                self.DisplayResults(iteration, LowestLoss)
            
            if iteration % 100 == 0:
                self.DisplayResults(iteration, LowestLoss) 

            BinaryLoss.backward(result, self.TrainY)
            Outputlayer.backward(BinaryLoss.dinputs)
            #Hiddenlayer2.backward(Outputlayer.dinputs)
            Hiddenlayer1.backward(Outputlayer.dinputs)

            Optimizer.UpdateParameters(Hiddenlayer1)
            #Optimizer.UpdateParameters(Hiddenlayer2)
            Optimizer.UpdateParameters(Outputlayer)
            
        # test
        Hiddenlayer1.forward(self.TestX)
        #Hiddenlayer2.forward(Hiddenlayer1.activation.outputs)
        Outputlayer.forward(Hiddenlayer1.activation.outputs)

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

        #input(f"{DM.Transpose(self.inputs)}\n\n{dvalues}")
        self.dweights = DM.DotProduct(DM.Transpose(self.inputs), dvalues)
        self.dbiases = [sum(x) for x in dvalues]

        self.dinputs = DM.DotProduct(dvalues, DM.Transpose(self.weights))

    def incrementVals(self, multiplier=0.05):
        FractionIncrease = [DM.Multiply(multiplier, np.random.randn(1, self.__NoOfNeurons).tolist()[0])
                       for sample in range(self.__NoOfInputs)]
        
        self.weights = [[a+b for a,b in zip(FractionIncrease[i], self.weights[i])] for i in range(self.__NoOfInputs)]

        self.biases = [a+b for a,b in zip(self.biases, DM.Multiply(multiplier, 
                                                                   np.random.randn(1, self.__NoOfNeurons).tolist()[0]))]
        
    