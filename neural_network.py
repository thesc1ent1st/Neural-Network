import numpy as np
from neuron import Neuron


class NeuralNetwork:
    """ combines neurons into a neural network """

    def __init__(self, weight_vect, bias):
        self.weights = weight_vect
        self.bias = bias
