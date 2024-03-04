from Scripts.NeuralNetwork.Layer import ReLU
from Scripts.NeuralNetwork.Layer import Sigmoid

import unittest

class TestRectifiedLinearUnit(unittest.TestCase):    
    def setUp(self):
        self.relu = ReLU()
    
    def test_reluForward(self):
        inputs = [[-1, 0, 1], [2, -2, 3]]
        self.relu.forward(inputs)
        expected = [[0, 0, 1], [2, 0, 3]]
        self.assertEqual(self.relu.outputs, expected)

    def test_reluBackward(self):
        inputs = [[-1, 0, 1], [2, -2, 3]]
        self.relu.forward(inputs)
        dvalues = None
        self.relu.backward(dvalues)
        expected = [[0, 0, 1], [1, 0, 1]]
        self.assertEqual(self.relu.dinputs, expected)

class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.sigmoid = Sigmoid()

    def test_sigmoidForward(self):
        inputs = [[-1], [0], [1]]
        self.sigmoid.forward(inputs)
        expected = [0.2689414213699951, 0.5, 0.7310585786300049]
        self.assertAlmostEqual(self.sigmoid.outputs, expected)

    def test_sigmoidBackward(self):
        inputs = [[-1], [0], [1]]
        self.sigmoid.forward(inputs)
        dvalues = [-3.7182818284590455, -2.0, -1.3678794411714423]
        self.sigmoid.backward(dvalues)
        expected = [[-0.7310585786300049], [-0.5], [-0.2689414213699951]]
        self.assertAlmostEqual(self.sigmoid.dinputs, expected)