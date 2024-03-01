from Scripts.LossAndOptimiser.Loss import BinaryCrossEntropy

import unittest

class TestBinaryCrossEntropy(unittest.TestCase):
    def setUp(self):
        self.LossFunction = BinaryCrossEntropy(regStr=0.001)

    def test_forward(self):
        predictions = [0.2, 0.7, 0.9]
        TrueValues = [0, 1, 1]
        self.LossFunction.forward(predictions, TrueValues)
        ExpectedLoss = -((0 * 0.2 + (1 - 0) * 0.7 + (1 - 0) * 0.9) / 3)
        self.assertAlmostEqual(self.LossFunction.getLoss(), ExpectedLoss, places=7)

    def test_backward(self):
        predictions = [0.2, 0.7, 0.9]
        TrueValues = [0, 1, 1]
        self.LossFunction.backward(predictions, TrueValues)
        ExpectedGradients = [(0.2 - 0) / ((1 - 0.2) * 0.2),
                              (0.7 - 1) / ((1 - 0.7) * 0.7),
                              (0.9 - 1) / ((1 - 0.9) * 0.9)]
        self.assertListEqual(self.LossFunction.dinputs, ExpectedGradients)

    def test_calcRegularisationLoss(self):
        layer_weights = [[0.1, 0.2], [0.3, 0.4]]
        self.LossFunction.calcRegularisationLoss(layer_weights)
        ExpectedLoss = 0.5 * 0.001 * (0.1**2 + 0.2**2 + 0.3**2 + 0.4**2)
        self.assertAlmostEqual(self.LossFunction.getLoss(), ExpectedLoss, places=7)

if __name__ == '__main__':
    unittest.main()