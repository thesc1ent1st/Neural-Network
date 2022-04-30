import numpy as np


class Neuron:
    """activates neurons for a neural network"""

    def __init__(self, vect_weights, bias):
        """
            vect_weights: an np.array([w1, ... , wn])
            bias: a real number representing the bias
        """
        self.vect_weights = vect_weights
        self.bias = bias

    def activation(self, vect_inputs):
        """
            vect_inputs: an np.array([x1, ... , xn])

            1) Weight our inputs.
            2) Add our bias.
            3) Use our sigmoid activation function.
        """
        weighted_sum = self._calculate_sum(vect_inputs)
        return self._sigmoid(weighted_sum)

    def _calculate_sum(self, vect_inputs):
        weighted = np.dot(vect_inputs, self.vect_weights)
        return weighted + self.bias

    def _sigmoid(self, weighted_sum):
        denom = (1 + (np.exp(-weighted_sum)))
        return 1 / denom


if __name__ == "__main__":
    inputs = np.array([2, 3])
    weights = np.array([0, 1])
    bias = 4

    neuron = Neuron(weights, bias)
    v1 = neuron.activation(inputs)
    print(f"The value is {v1}")
