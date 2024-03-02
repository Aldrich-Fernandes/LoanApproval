from Scripts.LossAndOptimiser.Optimiser import OptimiserSGD

import unittest

class TestOptimiserSGD(unittest.TestCase):
    def setUp(self):
        self.Optimiser = OptimiserSGD()

    def test_LearningRateDecay():
        pass # test both linear and exponential decay

    def test_ParameterAdjustment():
        pass
