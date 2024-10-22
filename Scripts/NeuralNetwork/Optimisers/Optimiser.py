from Scripts import DataMethod as DM

from math import exp

'''
Optimiser

Improve the accuracy of the model.

Does this by adjusting the weights and biases of layers by adding/subtracting a small amount to the weights 
depending on their impact on the model and its output which is calculated in the backpass (utilises the 
dvalues).

'''
class OptimiserSGD:
    def __init__(self, InitialLearningRate=1e-4, decay=5e-5, momentum=0.95):
        self.__InitialLearningRate = InitialLearningRate          # Starting Learning rate
        self.__minimumLearningRate = InitialLearningRate * 0.001  # Lower bound Learning rate
        self.__decay = decay                                      # Rate at which Learning rate decreases
        self.__momentum = momentum                                # Promotes adjustment in one direction
        self.activeLearningRate = InitialLearningRate             # How much to adjust/step. 

    # Gradually decreases the learning rate to avoid overshooting the optimal parameters
    # If it is too high it will overshoot the optimal but if too low the mode won't train properly.
    def adjustLearningRate(self, iter): 
        if self.__decay != 0:
            newLearningRate = self.__InitialLearningRate / (1 + self.__decay * iter)

            self.activeLearningRate = max(newLearningRate, self.__minimumLearningRate)

    # Function to update the parameters of a neural network layer using SGD with momentum
    def UpdateParameters(self, layer): 

        # Amount to increment the weights and biases
        weightUpdate = DM.Multiply(self.activeLearningRate, layer.dweights)
        biasesUpdate = DM.Multiply(self.activeLearningRate, layer.dbiases)

        weights, biases = layer.getWeightsAndBiases()

        if self.__momentum != 0:

            # Amount to added to the weights and biases to reduce fluctuations in accuracy and loss
            prevWeightsVelocity, prevBiasesVelocity = layer.getVelocities()

            # New weight velocity = momentum * Velocity - activeLearningRate * dweights
            newWeightsVelocity = [[a - b for a, b in zip(velocityRow, dweightsRow)] 
                                        for velocityRow, dweightsRow in zip(
                                            DM.Multiply(self.__momentum, prevWeightsVelocity), weightUpdate)]

            # New bias velocity = momentum * Velocity - activeLearningRate * dbiases
            newBiasesVelocity = [a - b for a, b in zip(
                                        DM.Multiply(self.__momentum, prevBiasesVelocity), biasesUpdate)]

            # Updates velocities for the layer to be used in the next loop
            layer.setVelocities(newWeightsVelocity, newBiasesVelocity)

            # Final (optimising) updates to the weights and biases of the layer for this loop
            weights = [[a + b for a, b in zip(weights[x], newWeightsVelocity[x])] 
                       for x in range(len(weights))]
            biases = [a + b for a, b in zip(biases, newBiasesVelocity)]
        else:
            weights = [[a - b for a, b in zip(weights[x], weightUpdate[x])] 
                       for x in range(len(weights))]
            biases = [a - b for a, b in zip(biases, biasesUpdate)]

        layer.setWeightsAndBiases(weights, biases)