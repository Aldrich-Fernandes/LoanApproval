from Scripts.NeuralNetwork.Optimisers import OptimiserSGD

import unittest

class TestOptimiserSGD(unittest.TestCase):
    def setUp(self):
        self.Optimiser = OptimiserSGD()

    def test_LearningRateDecay(self):
        pass # test both linear and exponential decay

    def test_ParameterAdjustment(self):
        pass
