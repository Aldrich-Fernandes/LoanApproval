from Scripts.Activations import ReLU, Sigmoid

import unittest

class TestActivations(unittest.TestCase):
    
    def test_reluForward(self):
        relu = ReLU()
        inputs = [[-1, 0, 1], [2, -2, 3]]
        relu.forward(inputs)
        expected = [[0, 0, 1], [2, 0, 3]]
        self.assertEqual(relu.outputs, expected)

    def test_reluBackward(self):
        relu = ReLU()
        inputs = [[-1, 0, 1], [2, -2, 3]]
        relu.forward(inputs)
        dvalues = None
        relu.backward(dvalues)
        expected = [[0, 0, 1], [1, 0, 1]]
        self.assertEqual(relu.dinputs, expected)

    def test_sigmoidForward(self):
        sigmoid = Sigmoid()
        inputs = [[-1], [0], [1]]
        sigmoid.forward(inputs)
        expected = [0.2689414213699951, 0.5, 0.7310585786300049]
        self.assertAlmostEqual(sigmoid.outputs, expected)

    def test_sigmoidBackward(self):
        sigmoid = Sigmoid()
        inputs = [[-1], [0], [1]]
        sigmoid.forward(inputs)
        dvalues = [-3.7182818284590455, -2.0, -1.3678794411714423]
        sigmoid.backward(dvalues)
        expected = [[-0.7310585786300049], [-0.5], [-0.2689414213699951]]
        self.assertAlmostEqual(sigmoid.dinputs, expected)

if __name__ == '__main__':
    unittest.main()
