from .ActivationABC import Activation

class ReLU(Activation): # Rectified Linear Unit
    # limits inputs between (>= 0)
    def forward(self, inputs):
        self.__inputs = inputs
        self.outputs = [[max(0, element) for element in entry]
                        for entry in inputs]

    # if input value was > 0 then gradient = 1
    def backward(self, _):
        self.dinputs = [[1 if element > 0 else 0 for element in sample] for sample in self.__inputs] 