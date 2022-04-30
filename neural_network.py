import numpy as np
from neuron import Neuron


class NeuralNetwork:
    """ combines neurons into a neural network """

    def __init__(self, weight_vect, bias):
        """ 2 inputs, 2 hidden, 1 output """
        self.weights = weight_vect
        self.bias = bias

        self.hidden_1 = Neuron(weight_vect, bias)
        self.hidden_2 = Neuron(weight_vect, bias)
        self.output_1 = Neuron(weight_vect, bias)

    def feed_forward(self, input_vect):
        output_hidden_1 = self.hidden_1.activation(input_vect)
        output_hidden_2 = self.hidden_2.activation(input_vect)
        hidden = np.array([output_hidden_1, output_hidden_2])

        output_output_1 = self.output_1.activation(hidden)

        return output_output_1


if __name__ == "__main__":
    bias = 0
    weight = np.array([0, 1])
    input = np.array([2, 3])

    neural_network = NeuralNetwork(weight, bias)
    output = neural_network.feed_forward(input)

    print(f"\noutput: {output}")
