from DataHandle import DataMethod, ShuffleData
from ActivationLossAndOptimisers import BinaryCrossEntropy, OptimiserSGD

import matplotlib.pyplot as plt

DM = DataMethod()

class Model:
    def __init__(self, Epochs=25, regularisationStrenght=0.001):                            # Creates a blank model
        self.Accuracy = 0.0
        self.__regularisationStrenght = regularisationStrenght
        self.__Epochs = Epochs
        
        self.__Layers = []

        # For backpass
        self.__LossFunction = BinaryCrossEntropy(self.__regularisationStrenght)                 # Loss function
        self.__Optimiser = OptimiserSGD(mode="Exponential")

    def add(self, layer):
        self.__Layers.append(layer)

    def configOptimizer(self, InitialLearningRate=1e-4, decay=5e-5, momentum=0.95, mode="Linear"):
        self.__Optimiser = OptimiserSGD(InitialLearningRate, decay, momentum, mode)

    def __forward(self, data):                                            # Forward progragation through layers
        for x, layer in enumerate(self.__Layers):
            if x == 0:
                layer.forward(data)
            else:
                layer.forward(self.__Layers[x-1].activation.outputs)

        return self.__Layers[-1].activation.outputs

    def train(self, X, Y, batch=32, show=False, canGraph=False):        # Fits the model based on input data
        losses = []
        accuracies = []
        lrs = []
        sampleSize = len(Y)

        # Epochs - How many times the model will see the data
        for iteration in range(self.__Epochs):
            accHold = []
            lossHold = []
            learningRateHold = []
            
            ShuffleData(X, Y)                                                                       # Improves generalisation
        
            for i in range(0, sampleSize, batch):                                                   # Reduces overfitting
                xBatch = X[i:i+batch] 
                yBatch = Y[i:i+batch]

                # Forward Pass
                result = self.__forward(xBatch)

                self.__LossFunction.forward(result, yBatch)
                self.__LossFunction.calcRegularisationLoss(self.__Layers[-1].getWeightsAndBiases()[0])

                accuracy = sum([1 for x,y in zip(result, yBatch) if round(x)==y]) / len(result)

                # Backward Pass 
                self.__LossFunction.backward(result, yBatch)
                
                for x, layer in enumerate(self.__Layers[::-1]):
                    if x == 0:
                        layer.backward(self.__LossFunction.dinputs)
                    else:
                        layer.backward(self.__Layers[-x].dinputs)

                self.__Optimiser.adjustLearningRate(iteration)
                for layer in self.__Layers:
                    self.__Optimiser.UpdateParameters(layer)


                # Tracking variables
                accHold.append(accuracy)
                lossHold.append(self.__LossFunction.getLoss())
                learningRateHold.append(self.__Optimiser.activeLearningRate)

            accuracies.append(sum(accHold) / (len(accHold)))
            losses.append(sum(lossHold) / (len(lossHold)))
            lrs.append(sum(learningRateHold) / (len(learningRateHold)))

            if show:
                self.__DisplayResults(iteration, loss=losses[-1], accuracy=accuracies[-1], learningRate=lrs[-1])
        
        if canGraph:
            self.__graph(accuracies, losses, lrs)
            input("Press ENTER to continue")

    def test(self, TestX, TestY, showTests=False):                      # Tests the model using data it has never seen
        result = self.__forward(TestX)
        
        if showTests:
            for x in range(len(result)):
                print(f"True: {TestY[x]} Predicted: {round(result[x])} Output: {result[x]}")
        
        self.Accuracy = sum([1 for x,y in zip(result, TestY) if round(x)==y]) / len(result)
        print(f"\nTest Accuracy: {self.Accuracy}")

    def Predict(self, UserData):                                        # Passes Userdata through model
        self.Result = round(self.__forward(UserData)[0], 4)

    def __graph(self, accuracies, losses, lrs, sep=True):                 # Displays obserable data
        X = [x for x in range(1, self.__Epochs+1)]
        if not sep:
            plt.plot(X, losses, label='Loss')
            plt.plot(X, accuracies, label='Accuracy')
            plt.legend()
        else:
            _, ax = plt.subplots(3, 1, figsize=(10, 8))
            ax[0].plot(X, losses, label='Loss')
            ax[1].plot(X, accuracies, label='Accuracy')
            ax[2].plot(X, lrs, label='Learning Rate')
            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
        plt.show(block=False)

    def __DisplayResults(self, iteration, loss, accuracy, learningRate):
        print(f"Iteration: {iteration} Loss: {round(loss, 5)} Accuracy: {round(accuracy, 5)} Lr: {learningRate}\n\n")

    def saveModel(self):  # save epochs regularisationStrenght, number of layers and thier weighs and biases
        pass

    def loadModel(self): # if 1 parameters: load data from file --- multiple parameters: from adjusted data
        pass