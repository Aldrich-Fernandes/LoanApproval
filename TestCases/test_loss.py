from Scripts.LossAndOptimiser import BinaryCrossEntropy

import unittest

class TestBinaryCrossEntropy(unittest.TestCase):

    def test_forward(self):
        LossFunction = BinaryCrossEntropy(regStr=0.001)
        predictions = [0.2, 0.7, 0.9]
        TrueValues = [0, 1, 1]
        LossFunction.forward(predictions, TrueValues)
        ExpectedLoss = -((0 * 0.2 + (1 - 0) * 0.7 + (1 - 0) * 0.9) / 3)
        self.assertAlmostEqual(LossFunction.getLoss(), ExpectedLoss, places=7)

    def test_backward(self):
        LossFunction = BinaryCrossEntropy(regStr=0.001)
        predictions = [0.2, 0.7, 0.9]
        TrueValues = [0, 1, 1]
        LossFunction.backward(predictions, TrueValues)
        ExpectedGradients = [(0.2 - 0) / ((1 - 0.2) * 0.2),
                              (0.7 - 1) / ((1 - 0.7) * 0.7),
                              (0.9 - 1) / ((1 - 0.9) * 0.9)]
        self.assertListEqual(LossFunction.dinputs, ExpectedGradients)

    def test_calcRegularisationLoss(self):
        LossFunction = BinaryCrossEntropy(regStr=0.001)
        layer_weights = [[0.1, 0.2], [0.3, 0.4]]
        LossFunction.calcRegularisationLoss(layer_weights)
        ExpectedLoss = 0.5 * 0.001 * (0.1**2 + 0.2**2 + 0.3**2 + 0.4**2)
        self.assertAlmostEqual(LossFunction.getLoss(), ExpectedLoss, places=7)

if __name__ == '__main__':
    unittest.main()