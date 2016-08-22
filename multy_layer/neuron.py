import abc
import math

import numpy as np


class Neuron(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, ntype, inputn, hidden, shape=(90000, 4)):
        self.type = ntype
        self.inputn = inputn
        self.hidden = hidden
        self.shape = shape
        self.__weights = None

    def _nguyen_widerow(self, weights):
        sc_factor = .7 * pow(self.hidden, 1.0/self.inputn)
        numerator = sc_factor * weights
        denominator = np.power(np.sum(np.squere(weights), axis=1), .5)
        return numerator / denominator

    def weights(self):
        if not self.__weights:
            randomw = np.random.uniform(-0.5, 0.5, self.shape)
            slices = (randomw[i] for i in range(self.shape[0]))
            self.__weights = map(self._nguyen_widerow, slices)
        return self.__weights

    def activation_function(self, input_):
        weighted = np.multiply(input_, self.weights)
        return 1. / (1 + math.exp(-sum(weighted)))

    @abc.abstractmethod
    def perceive(self, input_):
        # Different types of neurons perceive different types of data.
        raise NotImplementedError

    def learn(self, input_, correct):
        raise NotImplementedError

    @abc.abstractmethod
    def output(self):
        # Different types of neurons implement different behavior.
        # Input neuron just returns perceived information to a hidden levels of
        # neurons.
        # Hidden neuron returns a weighted signals to a next level of hidden
        # neurons or an output level.
        # Output neuron returns a decision.
        #return self.activation_function(input_) > .5
        raise NotImplementedError


class InputNeuron(object):
    def __init__(self, inputn, hidden, shape=(90000, 4)):
        self.inputn = inputn
        self.hidden = hidden
        self.shape = shape

    def perceive(self, input_):
        # Input neuron just perceives: determines background and returns
        # difference between object and background.
        raise NotImplementedError


class HiddenNeuron(Neuron):
    pass


class OutputNeuron(Neuron):
    pass


class Network(object):
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.input_neurons = map(self.create_input, xrange(self.layer_size))
        self.hidden_neurons = map(self.create_hidden, xrange(self.layer_size))
        self.outpur_neurons = map(self.create_output, xrange(self.layer_size))

    def create_input(self, idx):
        raise NotImplementedError

    def create_hidden(self, idx):
        raise NotImplementedError

    def create_output(self, idx):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def recognise(self):
        raise NotImplementedError
