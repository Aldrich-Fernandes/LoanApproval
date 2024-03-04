from Scripts.NeuralNetwork.LossFunctions import BinaryCrossEntropy

import unittest

class TestBinaryCrossEntropy(unittest.TestCase):
    def setUp(self):
        self.LossFunction = BinaryCrossEntropy(regStr=0.001)

    def test_forward(self):
        predictions = [0.2, 0.7, 0.9]
        TrueValues = [0, 1, 1]
        self.LossFunction.forward(predictions, TrueValues)
        ExpectedLoss = 0.22839 
        self.assertAlmostEqual(self.LossFunction.getLoss(), ExpectedLoss, places=5)

    def test_backward(self):
        predictions = [0.2, 0.7, 0.9]
        TrueValues = [0, 1, 1]
        self.LossFunction.backward(predictions, TrueValues)
        ExpectedGradients = [1.25, -1.42857142857, -1.11111111111]
        for result, expected in zip(self.LossFunction.dinputs, ExpectedGradients):
            self.assertAlmostEqual(result, expected)

    def test_calcRegularisationLoss(self):
        layer_weights = [[0.1, 0.2], [0.3, 0.4]]
        self.LossFunction.calcRegularisationLoss(layer_weights)
        ExpectedLoss = 0.00015
        self.assertAlmostEqual(self.LossFunction.getLoss(), ExpectedLoss, places=5)