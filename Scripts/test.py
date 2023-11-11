import numpy as np

class OptimizerSGD:
    def __init__(self, InitialLearningRate=1e-3, decay=1e-4, minimumLearningRate=1e-5, momentum=0.9):
        self.InitialLearningRate = InitialLearningRate
        self.minimumLearningRate = minimumLearningRate
        self.decay = decay
        self.momentum = momentum

        self.activeLearningRate = InitialLearningRate
        self.velocity_weights = None
        self.velocity_biases = None

    def adjustLearningRate(self, iter):
        if self.decay != 0: 
            # Linear 
            self.activeLearningRate = max(self.InitialLearningRate / (1 + self.decay * iter), self.minimumLearningRate)

    def UpdateParameters(self, layer):
        if self.velocity_weights is None:
            # Initialize velocity with zeros at the first iteration
            self.velocity_weights = np.zeros_like(layer.weights)
            self.velocity_biases = np.zeros_like(layer.biases)

        # Compute the adjusted gradients
        adjusted_dweights = [DM.Multiply(self.activeLearningRate, sample) for sample in layer.dweights]
        adjusted_dbiases = DM.Multiply(self.activeLearningRate, layer.dbiases)

        # Update velocity with momentum
        self.velocity_weights = np.multiply(self.momentum, self.velocity_weights) + adjusted_dweights
        self.velocity_biases = self.momentum * self.velocity_biases + adjusted_dbiases

        # Update weights and biases
        layer.weights = [[a - b for a, b in zip(layer.weights[x], self.velocity_weights[x])] for x in range(len(layer.weights))]
        layer.biases = [a - b for a, b in zip(layer.biases, self.velocity_biases)]
