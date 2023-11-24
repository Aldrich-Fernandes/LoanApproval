
from DataHandle import DataMethod
from ActivationLossAndOptimizers import ReLU, Sigmoid, BinaryCrossEntropy, OptimizerSGD

import matplotlib.pyplot as plt
from math import sqrt
from random import gauss, shuffle

DM = DataMethod()

class NeuralNetwork:
    def __init__(self, Epochs=35):
        self.Accuracy = 0.0
        self.loss = 9999999

        self.Epochs = Epochs
        self.losses = []
        self.Accuracies = []
        self.lrs = []

        self.Hiddenlayer = Layer(11, 8, ReLU())
        self.Outputlayer = Layer(8, 1, Sigmoid())

        # Currently overfitting
    def train(self, X, Y, show=False):

        # For backpass
        BinaryLoss = BinaryCrossEntropy() # Loss function
        Optimizer = OptimizerSGD(momentum=0.05)          # Optimizer

        # Epochs
        for iteration in range(self.Epochs):
            # Forward Pass

            a = list(zip(X, Y))
            shuffle(a)
            X, Y = zip(*a)
            X, Y = list(X), list(Y)
            
            self.Hiddenlayer.forward(X)
            self.Outputlayer.forward(self.Hiddenlayer.activation.outputs)

            result = self.Outputlayer.activation.outputs.copy()
            BinaryLoss.forward(result, Y)

            self.loss = BinaryLoss.SampleLoss

            self.Accuracy = sum([1 for x,y in zip(result, Y) if round(x)==y]) / len(result)

            self.losses.append(self.loss)
            self.Accuracies.append(self.Accuracy)

            # Backward Pass ---- Breaks here
            BinaryLoss.backward(result, Y)
            self.Outputlayer.backward(BinaryLoss.dinputs)
            self.Hiddenlayer.backward(self.Outputlayer.dinputs)

            Optimizer.adjustLearningRate(iteration)
            self.lrs.append(Optimizer.activeLearningRate)
            Optimizer.UpdateParameters(self.Hiddenlayer)
            Optimizer.UpdateParameters(self.Outputlayer)

            if show:
                self.DisplayResults(iteration, Optimizer.activeLearningRate)

    def graph(self, sep=False):
        X = [x for x in range(1, self.Epochs+1)]
        if not sep:
            plt.plot(X, self.losses, label='Loss')
            plt.plot(X, self.Accuracies, label='Accuracy')
            plt.legend()
        else:
            _, ax = plt.subplots(3, 1, figsize=(10, 8))
            ax[0].plot(X, self.losses, label='Loss')
            ax[1].plot(X, self.Accuracies, label='Accuracy')
            ax[2].plot(X, self.lrs, label='Learning Rate')
            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
        plt.show(block=False)

    def test(self, TestX, TestY, showTests=False):
        input(self.Outputlayer.activation.outputs)
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

        self.Result = round(self.Outputlayer.activation.outputs[0], 4)

    def DisplayResults(self, iteration, Lr):
        print(f"Iteration: {iteration} Loss: {round(self.loss, 5)} Accuracy: {round(self.Accuracy, 5)} Lr: {Lr}\n\n")

class Layer:
    def __init__(self, NoOfInputs, NoOfNeurons, activation):
        self.activation = activation

        # Xavier/Glorot weight initialization
        if self.activation.getID() == "ReLU":
            Numerator = 2
        elif self.activation.getID() == "Sigmoid":
            Numerator = 1

        scale = sqrt(Numerator / (NoOfInputs+NoOfNeurons))
        self.__weights = [[gauss(0, scale) for _ in range(NoOfNeurons)] for _ in range(NoOfInputs)]
        self.__biases = [0.0 for x in range(NoOfNeurons)]

        # Velocity for use with optimizer 
        self.__weightsVelocity = [[0.0 for _ in range(NoOfNeurons)] for _ in range(NoOfInputs)]
        self.__biasesVelocity = [0.0 for _ in range(NoOfNeurons)]

    def forward(self, inputs):
        self.inputs = inputs.copy() # (90x11)

        self.output = [[a+b for a,b in zip(sample, self.__biases)] for sample in DM.DotProduct(inputs, self.__weights)] # add biases  -- (10x7)/ (samplesize x NOofNeurons)        

        self.activation.forward(self.output)

    def backward(self, dvalues):
        self.activation.backward(dvalues)
        dvalues = self.activation.dinputs.copy()

        self.dinputs = DM.DotProduct(dvalues, DM.Transpose(self.__weights))

        self.dweights = DM.DotProduct(DM.Transpose(self.inputs), dvalues) # breaks here

        # result = np.sum(dvalues, axis=0, keepdims=True)
        self.dbiases = [sum(x) for x in DM.Transpose(dvalues)] ## wrong - hidden suold hacve 7 but gets 180

    def getVelocites(self):
        return self.__weightsVelocity, self.__biasesVelocity
    
    def setVelocities(self, veloWeights, veloBiases):
        self.__weightsVelocity, self.__biasesVelocity = veloWeights, veloBiases

    def getWeightsAndBaises(self):
        return self.__weights, self.__biases
    
    def setWeightsAndBaises(self, weights, biases):
        self.__weights, self.__biases = weights, biases
        
