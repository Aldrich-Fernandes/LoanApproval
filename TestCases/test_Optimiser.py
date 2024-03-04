from Scripts.NeuralNetwork.Optimisers import OptimiserSGD

import unittest

class TestOptimiser(unittest.TestCase):
    def test_LearningRateDecrease(self):
        initalLr = 0.1
        optimsier = OptimiserSGD(InitialLearningRate=initalLr, decay=0.01)

        for i in range(20):
            optimsier.adjustLearningRate(i)
            if optimsier.activeLearningRate == 0.0001:
                self.assertLessEqual(optimsier.activeLearningRate, 0.0001)
                break

            self.assertLessEqual(optimsier.activeLearningRate, initalLr)