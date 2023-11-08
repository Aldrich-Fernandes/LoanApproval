import numpy as np

# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)


        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                             self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                            self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


# Dropout
class Layer_Dropout:

    # Init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs):
        # Save input values
        self.inputs = inputs
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,
                           size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)


# Sigmoid activation
class Activation_Sigmoid:

    # Forward pass
    def forward(self, inputs):
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output


# SGD optimizer
class Optimizer_SGD:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))


    # Update parameters
    def update_params(self, layer):

        # If we use momentum
        if self.momentum:

            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * \
                             layer.dweights
            bias_updates = -self.current_learning_rate * \
                           layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1



# Adagrad optimizer
class Optimizer_Adagrad:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1



# RMSprop optimizer
class Optimizer_RMSprop:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1



# Adam optimizer
class Optimizer_Adam:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                             self.epsilon)
        layer.biases += -self.current_learning_rate * \
                         bias_momentums_corrected / \
                         (np.sqrt(bias_cache_corrected) +
                             self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# Common loss class
class Loss:

    # Regularization loss calculation
    def regularization_loss(self, layer):

        # 0 by default
        regularization_loss = 0

        # L1 regularization - weights
        # calculate only when factor greater than 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * \
                                   np.sum(np.abs(layer.weights))

        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * \
                                   np.sum(layer.weights *
                                          layer.weights)

        # L1 regularization - biases
        # calculate only when factor greater than 0
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * \
                                   np.sum(np.abs(layer.biases))

        # L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * \
                                   np.sum(layer.biases *
                                          layer.biases)

        return regularization_loss

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)


        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Binary cross-entropy loss
class Loss_BinaryCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
# Create dataset
X = np.array([[0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.6522041396593705, 1.4220925199724657, -0.249497918356507, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.8043566126074282, -0.8687951893569025, -1.065987887227051, 0.2832752103051981, -2.2912878474779195, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, -2.0647416048350564, -0.36927447293799814, 0.2432729988821191, -0.1069418348589963, 0.2859053399192595, 0.2832752103051981, -2.2912878474779195, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, 0.4843221048378526, 2.70801280154532, 4.419772902388914, -0.8687951893569025, 3.4849398081169642, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, -2.0647416048350564, -0.36927447293799814, -0.00666735555615084, -0.21402751615555515, 0.24575009554857702, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, 0.4843221048378526, -0.36927447293799814, 0.2313059504480022, -0.8687951893569025, -0.6108951176926494, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, 2.70801280154532, -0.38106501370923646, -0.8687951893569025, -1.5344657382183466, -2.99158271229825, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, 0.4843221048378526, -0.36927447293799814, -0.4836397145730955, -0.6231907163334516, -0.42350397729613115, -3.482811400688767, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.45970561770486174, 0.5904470050275485, -0.3967338143823428, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, -1.5275252316519463, 1.5142305903535844, 0.4843221048378526, -0.36927447293799814, 1.5705896280604554, -0.8687951893569025, 0.6071472948847194, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.5123606308149761, 0.41836404453606346, -0.008566452132412082, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, -2.0647416048350564, -0.36927447293799814, -0.30174057837451873, -0.8687951893569025, -0.3967338143823428, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [-2.70801280154532, -1.5275252316519463, 0.31246028054915237, 0.4843221048378526, 2.70801280154532, 1.3849294194968704, -0.8687951893569025, 0.1386694438934237, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, -2.0647416048350564, -0.36927447293799814, -0.3072112290872579, -0.7398661601341798, -0.42350397729613115, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, 0.48774270260764985, -0.8687951893569025, 0.018203710781376243, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.19608863648474392, 0.4631162695555209, 0.0048186293244820814, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [-2.70801280154532, 0.6546536707079772, 0.31246028054915237, 0.4843221048378526, 2.70801280154532, 2.368278885111733, -0.8687951893569025, 1.9590405220310299, 0.2832752103051981, -2.2912878474779195, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, -2.0647416048350564, -0.36927447293799814, -0.9483031094863771, 0.6943360988227176, -0.6242801991495436, 0.2832752103051981, -2.2912878474779195, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, -2.0647416048350564, 2.70801280154532, -0.9226594342704123, 0.13013840482741504, -0.570739873321967, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, 0.4843221048378526, -0.36927447293799814, 0.3058435664090731, -0.3189821391178541, -1.5210806567614525, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [-2.70801280154532, -1.5275252316519463, 0.31246028054915237, 0.4843221048378526, -0.36927447293799814, -0.6747705738494195, -0.8687951893569025, 0.27252025846236533, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, 0.4843221048378526, 2.70801280154532, 2.0308081192696363, 1.1823484573682295, 0.27252025846236533, -2.1728682316473877, -2.2912878474779195, 0.13018891098082386], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.3328549043032227, -0.8687951893569025, -0.7982862580891676, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, 0.4843221048378526, -0.36927447293799814, -0.4850073772512803, 0.078993004805178, 0.0048186293244820814, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, -2.0647416048350564, -0.36927447293799814, -0.5055223174240521, -0.8687951893569025, -0.3565785700116603, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [-2.70801280154532, 0.6546536707079772, 1.5142305903535844, 0.4843221048378526, -0.36927447293799814, 3.5191670288042314, -0.8687951893569025, -0.9321370726581093, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [-2.70801280154532, -1.5275252316519463, 0.31246028054915237, -2.0647416048350564, -0.36927447293799814, 0.0111122592602514, -0.8687951893569025, -0.7849011766322735, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [-2.70801280154532, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.5721958729855605, -0.8687951893569025, -0.9187519912012152, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, 0.4843221048378526, -0.36927447293799814, -0.3954254718301767, 0.06354283188179388, 0.6205323763416136, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, 0.4843221048378526, -0.36927447293799814, -0.42414638807205723, 0.8626897072292479, 0.24575009554857702, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [-2.70801280154532, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.3636273145623804, -0.8687951893569025, -0.8518265839167444, 0.2832752103051981, -2.2912878474779195, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.6115161749833732, 0.20525821111007572, -0.3967338143823428, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, -2.0647416048350564, -0.36927447293799814, -0.6747705738494195, 0.03690460270354542, -0.4368890587530253, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, -2.0647416048350564, -0.36927447293799814, 0.5304821613009245, -0.003052741063827284, 0.6339174577985077, 1.920704171606922, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.7089621408040393, 1.1535791698557212, -0.2628829998134012, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.7346058160200041, 0.37787393618512577, 0.04497387369516457, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, 0.4843221048378526, -0.36927447293799814, -0.4303008701238888, -0.8687951893569025, -0.5841249547788611, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, 0.42448830374160346, 1.0843197739922752, 2.293667558453384, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.7147747071863246, 0.44500227371431195, -0.6644354435202261, -2.1728682316473877, -2.2912878474779195, 1.4320780207890629], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, 2.70801280154532, 0.5048384860849597, 1.4396737512301097, -0.2628829998134012, -2.1728682316473877, 0.4364357804719847, 0.13018891098082386], [-2.70801280154532, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.7352896473590964, -0.5997490746565929, -0.12903218524445956, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, -2.0647416048350564, -0.36927447293799814, -0.6747705738494195, 0.14931792983575395, -0.3164233256409778, 0.2832752103051981, -2.2912878474779195, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.09317201995133866, -0.8687951893569025, -0.2896531627271895, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.4839816302426417, 0.7577350842669489, -0.6778205249771202, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, -2.0647416048350564, -0.36927447293799814, 0.49902591970267435, -0.8687951893569025, 0.27252025846236533, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, 0.4843221048378526, -0.36927447293799814, 0.41252125530748657, -0.8687951893569025, 0.7008428650829786, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.15061385243509973, -0.8687951893569025, -0.23611283689961285, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, 0.4843221048378526, -0.36927447293799814, 0.472356497478071, 2.12800559319605, 2.3472078842809605, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.7988859618946891, -0.8687951893569025, -1.0258326428563684, 1.920704171606922, -2.2912878474779195, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.7298189966463573, 0.9069091676651403, -0.1424172667013537, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [-2.70801280154532, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.14138212935735242, -0.8687951893569025, -0.9187519912012152, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, 0.4843221048378526, -0.36927447293799814, -0.44124217154936707, 0.21538073819781015, 0.11189928097963539, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.08223071852586035, 0.4370108049608374, -0.3967338143823428, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.30686931341771173, 1.4226252845560305, 0.4331412359450953, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, 1.2854319596589272, 1.1290719990117326, 0.6339174577985077, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [-2.70801280154532, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, 2.70801280154532, 0.9804431824237196, -0.8687951893569025, 0.7276130279967669, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, 1.1486656918404485, -0.8687951893569025, 1.811804626005194, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, 0.4843221048378526, -0.36927447293799814, 0.32225551854729056, 5.151444604927251, 4.75652254652191, 0.2832752103051981, -2.2912878474779195, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, 0.4843221048378526, 2.70801280154532, -0.42995895445434257, 0.9958808531204902, -0.6912056064340144, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.8347871071970397, -0.021166736905036242, -0.5841249547788611, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, 0.4843221048378526, -0.36927447293799814, 1.1961919699073698, -0.7409316893013098, 1.5173328339535226, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, 0.4843221048378526, -0.36927447293799814, 0.7869189134605722, -0.8687951893569025, 0.8079235167381319, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, 0.4843221048378526, -0.36927447293799814, -0.08223071852586035, 0.09604147147925701, 0.27252025846236533, 0.2832752103051981, -2.2912878474779195, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, -2.0647416048350564, -0.36927447293799814, -0.22446763705707826, -0.8687951893569025, -0.2093426739858245, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, 0.4843221048378526, -0.36927447293799814, -0.7120393818299551, 0.22550326528554457, -0.47704430312370777, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, 0.4843221048378526, -0.36927447293799814, -0.15984557551284706, -0.8687951893569025, -1.3336895163649343, -2.1728682316473877, 0.4364357804719847, 1.4320780207890629], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, 0.1099258877591023, -0.8687951893569025, -0.5038144660374961, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, 0.4843221048378526, -0.36927447293799814, -1.1196028599290218, -0.29021285160534577, -1.6415463898734999, -2.99158271229825, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.13933063534007523, -0.8687951893569025, -0.5573547918650728, 0.2832752103051981, -2.2912878474779195, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, 0.4843221048378526, -0.36927447293799814, 2.111842132952085, -0.8687951893569025, 2.307052639910278, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.4812463048862721, 0.23083091112119425, -0.47704430312370777, 0.2832752103051981, -2.2912878474779195, 0.13018891098082386], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.7089621408040393, -0.8687951893569025, -0.5841249547788611, 1.920704171606922, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, -2.0647416048350564, -0.36927447293799814, -0.32601659091229873, -0.8687951893569025, -1.5344657382183466, -2.99158271229825, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.5499713544650577, 2.1684957015469877, 0.20559485117789453, -3.8102971929491116, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, -1.5275252316519463, 0.31246028054915237, 0.4843221048378526, -0.36927447293799814, -0.06479301937900431, 0.08645170897508757, -0.30303824418408365, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.8515409750048034, 0.8094132488727509, -0.15580234815824787, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, 0.4843221048378526, -0.36927447293799814, 0.5099672211281526, 0.19886503610729608, 1.47717758958284, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, 2.70801280154532, 1.9976422993236553, -0.8687951893569025, 0.6339174577985077, 0.2832752103051981, -2.2912878474779195, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, -2.0647416048350564, -0.36927447293799814, -0.6566490433634712, -0.06964831400944844, -0.3565785700116603, -2.1728682316473877, -2.2912878474779195, 1.4320780207890629], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, 0.4843221048378526, 2.70801280154532, 4.086405124581372, -0.32857190162202354, 0.1386694438934237, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [0.36927447293799814, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.24464066156030387, -0.8687951893569025, -0.8518265839167444, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [-2.70801280154532, -1.5275252316519463, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.5085995584499678, -0.8687951893569025, -0.2628829998134012, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.6412628382338923, 0.21484797361424518, -0.23611283689961285, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, 0.4843221048378526, 2.70801280154532, -0.38414225473515223, 0.23882237987466878, 0.2992904213761537, 0.2832752103051981, 0.4364357804719847, -1.171700198827415], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.36807221826648095, 0.1775544527646973, -0.3164233256409778, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, 0.4843221048378526, -0.36927447293799814, -0.6522041396593705, 0.24095343820892867, -0.5975100362357553, 0.2832752103051981, 0.4364357804719847, 1.4320780207890629], [0.36927447293799814, 0.6546536707079772, 1.5142305903535844, 0.4843221048378526, -0.36927447293799814, -0.5130444621540684, 3.449794525020739, 0.8079235167381319, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, -2.0647416048350564, -0.36927447293799814, -0.7982021305555966, 0.47590261956108015, -0.15580234815824787, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, 0.31246028054915237, 0.4843221048378526, -0.36927447293799814, -0.23882809517801853, -0.8687951893569025, -0.9722923170287918, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386], [0.36927447293799814, 0.6546536707079772, -0.8893100292552798, 0.4843221048378526, -0.36927447293799814, -0.5267210889359163, -0.09149166193561219, -0.5975100362357553, 0.2832752103051981, 0.4364357804719847, 0.13018891098082386]])
y = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y = y.reshape(-1, 1)

# Create Dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(11, 64, weight_regularizer_l2=5e-4,
                            bias_regularizer_l2=5e-4)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 1 output value
dense2 = Layer_Dense(64, 1)

# Create Sigmoid activation:
activation2 = Activation_Sigmoid()

# Create loss function
loss_function = Loss_BinaryCrossentropy()

# Create optimizer
optimizer = Optimizer_Adam(decay=5e-7)

# Train in loop
for epoch in range(10001):

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function
    # of first layer as inputs
    dense2.forward(activation1.output)

    # Perform a forward pass through activation function
    # takes the output of second dense layer here
    activation2.forward(dense2.output)

    # Calculate the data loss
    data_loss = loss_function.calculate(activation2.output, y)
    # Calculate regularization penalty
    regularization_loss = \
        loss_function.regularization_loss(dense1) + \
        loss_function.regularization_loss(dense2)

    # Calculate overall loss
    loss = data_loss + regularization_loss

    # Calculate accuracy from output of activation2 and targets
    # Part in the brackets returns a binary mask - array consisting
    # of True/False values, multiplying it by 1 changes it into array
    # of 1s and 0s
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, '+
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


# Validate the model

# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=2)

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y_test = y_test.reshape(-1, 1)


# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)

# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through activation function
# takes the output of second dense layer here
activation2.forward(dense2.output)

# Calculate the data loss
loss = loss_function.calculate(activation2.output, y_test)

# Calculate accuracy from output of activation2 and targets
# Part in the brackets returns a binary mask - array consisting of
# True/False values, multiplying it by 1 changes it into array
# of 1s and 0s
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions==y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
