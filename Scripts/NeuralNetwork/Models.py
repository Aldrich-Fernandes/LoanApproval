from DataHandle import DataMethod as DM
from LossAndOptimiser import OptimiserSGD, BinaryCrossEntropy

import matplotlib.pyplot as plt

'''
Model - Logistic Regression

Acts as the brain of the model, collating and controlling all components needed to perform Logistic 
Regression.

'''
class LogisticRegression:
    # Creates a blank model
    def __init__(self, Epochs=25, regularisationStrength=0.001):
        # Tracking variables
        self.Accuracy = 0.0
        self.__regStr = regularisationStrength      # How strongly to penelise the model for strong weights
        self.__Epochs = Epochs                      # How many times the model will see the data
        
        # Layers
        self.__Layers = []

        # For backpass
        self.__LossFunction = BinaryCrossEntropy(self.__regStr)     # Calcuates the model's performance
        self.__Optimiser = OptimiserSGD()                           # Improves the model

    # Configuration Modules
        
    # Adds new layer to the model
    def addLayer(self, layer):
        self.__Layers.append(layer)

    # Resets layer with new, random weights
    def resetLayers(self):
        for layer in self.__Layers:
            layer.initialiseNewLayer() 

    # Resets Optimiser and update it's hyperparameters
    def configOptimiser(self, InitialLearningRate=1e-4, decay=5e-5, momentum=0.95):
        self.__Optimiser = OptimiserSGD(InitialLearningRate, decay, momentum)

    # Changes epochs for training
    def updateEpoch(self, epoch):
        self.__Epochs = epoch

    # Changes regularisation strenght
    def updateRegStr(self, regStr):
        self.__regStr = regStr
        self.__LossFunction.updateRegStr(regStr)

    # Used for forward progragation through layers
    def __forward(self, data):                                     
        for x, layer in enumerate(self.__Layers):
            if x == 0:
                layer.forward(data)
            else:       # Takes the activated output of the prior layer
                layer.forward(self.__Layers[x-1].activation.outputs)

        return self.__Layers[-1].activation.outputs

    # Trains the model based on input data
    def train(self, X, Y, batch=32, show=False, canGraph=False):
        # Data holders to used when plotting the graph and calculating values
        losses = []
        accuracies = []
        lrs = []
        sampleSize = len(Y)

        # Training loop
        for iteration in range(self.__Epochs):
            # dataholders for that Epoch (holds output of each batch)
            accHold = []
            lossHold = []
            learningRateHold = []
            
            DM.ShuffleData(X, Y)                        # Shuffling dataset - Improves generalisation
        
            # Using batchs - Reduces overfitting by passing smaller groups of data to the model at a time
            for i in range(0, sampleSize, batch):
                xBatch = X[i:i+batch] 
                yBatch = Y[i:i+batch]

                # Forward Pass
                result = self.__forward(xBatch)

                # Evaluating the performace of the model 
                self.__LossFunction.forward(result, yBatch)
                self.__LossFunction.calcRegularisationLoss(self.__Layers[-1].getWeightsAndBiases()[0])

                accuracy = sum([1 for x,y in zip(result, yBatch) if round(x)==y]) / len(result)

                # Backward Pass 

                # Calculating gradients (explained in Layer and Optimiser files)
                self.__LossFunction.backward(result, yBatch)
                
                for x, layer in enumerate(self.__Layers[::-1]):
                    if x == 0:
                        layer.backward(self.__LossFunction.dinputs)
                    else:
                        layer.backward(self.__Layers[-x].dinputs)

                # Optimising the layer parameters - changing weights and biases
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

            # Output evaluation of training loop
            if show:
                self.__DisplayResults(iteration, loss=losses[-1], accuracy=accuracies[-1], 
                                      learningRate=lrs[-1])
        
        # Visulaises the training outcomes
        if canGraph:
            self.__graph(accuracies, losses, lrs)

    # Tests the model using data it has never seen
    def test(self, TestX, TestY, showTests=False):
        result = self.__forward(TestX)
        
        if showTests:
            for x in range(len(result)):
                print(f"True: {TestY[x]} Predicted: {round(result[x])} Output: {result[x]}")
        
        self.Accuracy = sum([1 for x,y in zip(result, TestY) if round(x)==y]) / len(result)

    # Passes Userdata through model
    def Predict(self, UserData):
        self.Result = round(self.__forward(UserData)[0], 4)

    # Displays obserable data - used to intepret issues with the model and manually tune hyperparameters
    def __graph(self, accuracies, losses, lrs, sep=True):
        X = [x for x in range(1, self.__Epochs+1)]
        if not sep:     # All data on same graph
            plt.plot(X, losses, label='Loss')
            plt.plot(X, accuracies, label='Accuracy')
            plt.legend()
        else:           # Different data on seperate graphs
            _, ax = plt.subplots(3, 1, figsize=(10, 8))
            ax[0].plot(X, losses, label='Loss')
            ax[1].plot(X, accuracies, label='Accuracy')
            ax[2].plot(X, lrs, label='Learning Rate')
            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
        plt.show(block=False)

    # Outputs evaluation for that iteration
    def __DisplayResults(self, iter, loss, accu, Lr):
        print(f"Iteration: {iter} Loss: {round(loss, 5)} Accuracy: {round(accu, 5)} Lr: {Lr}\n\n")

    # Saves the model in a txt file
    def saveModel(self, filePath, ScalingData):
        try:
            # Save preprocess ting needed for encoding
            file = open(filePath,  "w")
            file.write(f"{ScalingData}\n")
            for layer in self.__Layers:
                weights, biases = layer.getWeightsAndBiases()
                file.write(f"{weights}\n")
                file.write(f"{biases}\n")
            return "Model Saved Successfully."
        except FileExistsError:
            return "Filename already used. Try again."

    # Loads a model from a txt file 
    def loadModel(self, filePath):
        self.resetLayers()
        try:
            file = open(filePath,  "r")
            scalingData = eval(file.readline().rstrip())
            for layer in self.__Layers:
                weights = eval(file.readline().rstrip())
                biases = eval(file.readline().rstrip())

                if len(layer.getWeightsAndBiases()[1]) != len(biases):
                    print("Layers dont match... cant load.")
                else:
                    layer.setWeightsAndBiases(weights, biases)

            return scalingData
        except FileNotFoundError:
            return -1