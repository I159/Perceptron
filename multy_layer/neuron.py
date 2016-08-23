import abc
import math
from pillow import Image
import unittest

import numpy as np


class Neuron(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, inputn, hidden, shape=(90000, 4)):
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

    def learn(self, input_, correct):
        raise NotImplementedError

    @abc.abstractmethod
    def perceive(self, input_):
        # Different types of neurons perceive different types of data.
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
        self.image_size = (300, 300)

    def _standardize_size(self, image):
        image.thumbnail(self.image_size, Image.ANTIALIAS)
        bordered = Image.new('RGBA', self.image_size, (255, 255, 255, 0))
        borders_size = (
            (self.image_size[0] - image.size[0]) / 2,
            (self.image_size[1] - image.size[1]) / 2)
        bordered.paste(image, borders_size)
        return bordered

    def _get_rgba(self, file_path):
        image = Image.open(file_path)
        standardized_image = self._standardize_size(image)
        return np.array(standardized_image.getdata())

    @staticmethod
    def _get_background(input_signal):
        view_shape = [('', input_signal.dtype)]*input_signal.shape[1]
        view = input_signal.view(view_shape)
        unique_a = np.unique(view, return_counts=True)
        max_count = unique_a[0][unique_a[1].argmax(axis=0)].tolist()
        return np.array(max_count, dtype=float)

    def perceive(self, file_path):
        rgba = self._get_rgba(file_path)
        background = self._get_background(rgba)
        diff = np.subtract(rgba, background)
        abs_diff = np.absolute(diff) / 256.0
        return abs_diff


class HiddenNeuron(Neuron):
    def perceive(self, input_):
        raise NotImplementedError

    def output(self):
        raise NotImplementedError


class OutputNeuron(Neuron):
    def __init__(self, init_data, *args):
        # Init output
        super(OutputNeuron, self).__init__(*args)

    def perceive(self, input_):
        raise NotImplementedError

    def output(self):
        raise NotImplementedError


class Network(object):
    def __init__(self, layer_size, output_init_data):
        self.layer_size = layer_size
        self.io_size = len(output_init_data)
        self.input_layer = map(self.create_input, xrange(self.io_size))
        self.hidden_layer = map(self.create_hidden, xrange(self.layer_size))
        self.output_layer = map(self.create_output, output_init_data)

    def create_input(self, idx):
        return InputNeuron(self.io_size, self.layer_size)

    def create_hidden(self, idx):
        return HiddenNeuron(self.io_size, self.layer_size)

    def create_output(self, init_data):
        return OutputNeuron(init_data, self.io_size, self.layer_size)

    def learn(self):
        raise NotImplementedError

    def recognise(self, file_path):
        input_ = (neuron.perceive for neuron in self.input_layer)
        hidden = (neuron.perceive(input_) for neuron in self.hidden_layer)
        return (neuron.perceive(hidden) for neuron in self.output_layer)
