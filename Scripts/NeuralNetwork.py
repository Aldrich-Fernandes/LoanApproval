from DataHandle import DataMethod, ShuffleData
from ActivationLossAndOptimizers import BinaryCrossEntropy, OptimizerSGD

import matplotlib.pyplot as plt

DM = DataMethod()

class Model:
    def __init__(self, Epochs=10, regularisationStrenght=0.001):                            # Creates a blank model
        self.Accuracy = 0.0
        self.regularisationStrenght = regularisationStrenght
        self.Epochs = Epochs
        
        self.Layers = []

        # For backpass
        self.LossFunction = BinaryCrossEntropy(self.regularisationStrenght)                 # Loss function
        self.Optimizer = OptimizerSGD(InitialLearningRate=1e-4, decay=5e-5, momentum=0.95)  # Optimizer

    def add(self, layer):
        self.Layers.append(layer)

    def forward(self, data):                                            # Forward progragation through layers
        for x, layer in enumerate(self.Layers):
            if x == 0:
                layer.forward(data)
            else:
                layer.forward(self.layers[x-1].activation.outputs)

        return self.Layers[-1].activation.outputs

    def train(self, X, Y, batch=16, show=False, canGraph=False):        # Fits the model based on input data
        losses = []
        accuracies = []
        lrs = []
        sampleSize = len(Y)

        # Epochs - How many times the model will see the data
        for iteration in range(self.Epochs):
            accHold = []
            lossHold = []
            learningRateHold = []
            
            ShuffleData(X, Y)                                                                       # Improves generalisation
        
            for i in range(0, sampleSize, batch):                                                   # Reduces overfitting
                xBatch = X[i:i+batch] 
                yBatch = Y[i:i+batch]

                # Forward Pass
                result = self.forward(xBatch)

                self.LossFunction.forward(result, yBatch)
                self.LossFunction.calcregularisationLoss(self.Layers[-1].getWeightsAndBiases()[0])

                accuracy = sum([1 for x,y in zip(result, yBatch) if round(x)==y]) / len(result)

                # Backward Pass 
                self.LossFunction.backward(result, yBatch)
                
                for x, layer in enumerate(self.Layers[::-1]):
                    if x == 0:
                        layer.backward(self.LossFunction.dinputs)
                    else:
                        layer.backward(self.Layers[-x].dinputs)

                self.Optimizer.adjustLearningRate(iteration)
                for layer in self.Layers:
                    self.Optimizer.UpdateParameters(layer)


                # Tracking variables
                accHold.append(accuracy)
                lossHold.append(self.LossFunction.getLoss())
                learningRateHold.append(self.Optimizer.activeLearningRate)

            accuracies.append(sum(accHold) / (len(accHold)))
            losses.append(sum(lossHold) / (len(lossHold)))
            lrs.append(sum(learningRateHold) / (len(learningRateHold)))

            if show:
                self.DisplayResults(iteration, loss=losses[-1], accuracy=accuracies[-1], learningRate=lrs[-1])
        
        if canGraph:
            self.graph(accuracies, losses, lrs)

    def test(self, TestX, TestY, showTests=False):                      # Tests the model using data it has never seen
        result = self.forward(TestX)
        
        if showTests:
            for x in range(len(result)):
                print(f"True: {TestY[x]} Predicted: {round(result[x])} Output: {result[x]}")
        
        self.Accuracy = sum([1 for x,y in zip(result, TestY) if round(x)==y]) / len(result)
        print(f"\nTest Accuracy: {self.Accuracy}")

    def Predict(self, UserData):                                        # Passes Userdata through model
        self.Result = round(self.forward(UserData)[0], 4)

    def graph(self, accuracies, losses, lrs, sep=True):                 # Displays obserable data
        X = [x for x in range(1, self.Epochs+1)]
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

    def DisplayResults(self, iteration, loss, accuracy, learningRate):
        print(f"Iteration: {iteration} Loss: {round(loss, 5)} Accuracy: {round(accuracy, 5)} Lr: {learningRate}\n\n")

    def saveModel(self):
        pass

    def loadModel(self):
        pass