from Scripts.Activations import ReLU, Sigmoid

import unittest

class TestActivations(unittest.TestCase):
    
    def reluForward(self):
        relu = ReLU()
        inputs = [[-1, 0, 1], [2, -2, 3]]
        relu.forward(inputs)
        expected = [[0, 0, 1], [2, 0, 3]]
        self.assertEqual(relu.outputs, expected)

    def reluBackward(self):
        relu = ReLU()
        inputs = [[-1, 0, 1], [2, -2, 3]]
        relu.forward(inputs)
        dvalues = [[1, 2, 3], [4, 5, 6]]
        relu.backward(dvalues)
        expected = [[0, 0, 3], [4, 0, 6]]
        self.assertEqual(relu.dinputs, expected)

    def sigmoidForward(self):
        sigmoid = Sigmoid()
        inputs = [[-1], [0], [1]]
        sigmoid.forward(inputs)
        expected = [0.2689414213699951, 0.5, 0.7310585786300049]
        self.assertEqual(sigmoid.outputs, expected)

    def sigmoidBackward(self):
        sigmoid = Sigmoid()
        inputs = [[-1], [0], [1]]
        sigmoid.forward(inputs)
        dvalues = [[1], [2], [3]]
        sigmoid.backward(dvalues)
        expected = [[0.19661193324148185], [0.0], [0.19661193324148185]]
        self.assertEqual(sigmoid.dinputs, expected)

if __name__ == '__main__':
    unittest.main()
