import numpy as np

from DataHandle import DataMethod
from ActivationLossAndOptimizers import ReLU, Sigmoid, BinaryCrossEntropy, OptimizerSGD

import matplotlib.pyplot as plt

DM = DataMethod()

class NeuralNetwork:
    def __init__(self, Epochs=75):
        self.Accuracy = 0.0
        self.LowestLoss = 9999999
        
        self.Epochs = Epochs
        self.losses = []
        self.Accuracies = []
        self.lrs = []

        self.Hiddenlayer = Layer(11, 7, ReLU())
        self.Outputlayer = Layer(7, 1, Sigmoid())
        
        # Currently overfitting
    def train(self, TrainX, TrainY, show=False):
        X, Y = TrainX, TrainY

        # For backpass
        BinaryLoss = BinaryCrossEntropy()
        Optimizer = OptimizerSGD()

        # Epochs
        for iteration in range(self.Epochs):
        
            self.Hiddenlayer.forward(X)
            self.Outputlayer.forward(self.Hiddenlayer.activation.outputs)

            result = self.Outputlayer.activation.outputs.copy()
            loss = BinaryLoss.forward(result, Y)

            self.Accuracy = sum([1 for x,y in zip(result, Y) if round(x)==y]) / len(result)

            self.losses.append(loss)
            self.Accuracies.append(self.Accuracy)
            
            if loss < self.LowestLoss:
                self.LowestLoss = loss

            BinaryLoss.backward(result, Y)
            self.Outputlayer.backward(BinaryLoss.dinputs)
            self.Hiddenlayer.backward(self.Outputlayer.dinputs)

            Optimizer.UpdateParameters(self.Hiddenlayer)
            Optimizer.UpdateParameters(self.Outputlayer)
            
            if show:
                self.DisplayResults(iteration+1)
                

    def graph(self, sep=False):
        X = [x for x in range(1, self.Epochs+1)]
        if not sep:
            plt.plot(X, self.losses, label='Loss')
            plt.plot(X, self.Accuracies, label='Accuracy')
            plt.legend()
        else:
            fig, ax = plt.subplots(3, 1, figsize=(10, 8))
            ax[0].plot(X, self.losses, label='Loss')
            ax[1].plot(X, self.Accuracies, label='Accuracy')
            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
        plt.show(block=False)
        

    def test(self, TestX, TestY, showTests=False):
        self.Hiddenlayer.forward(TestX)
        self.Outputlayer.forward(self.Hiddenlayer.activation.outputs)

        result = self.Outputlayer.activation.outputs.copy()
        if showTests:
            for x in range(len(result)):
                print(f"True: {TestY[x]} Predicted: {round(result[x])} Output: {result[x]}")
        
        print("Test Accuracy: ", str(sum([1 for x,y in zip(result, TestY) if round(x)==y]) / len(result)))

    def Predict(self, UserData):
        self.Hiddenlayer.forward([UserData])
        self.Outputlayer.forward(self.Hiddenlayer.activation.outputs)

        result = round(self.Outputlayer.activation.outputs[0], 4)
        if round(result) == 1:
            print(f"You a likely to be approved. Confidence = {result * 100}%")
        else:
            print(f"You a unlikely to be approved. Confidence = {(1-result) * 100}%")
    
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