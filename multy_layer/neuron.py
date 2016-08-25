import abc
import math
from pillow import Image
import unittest

import numpy as np


class Neuron(object):
    __metaclass__ = abc.ABCMeta

    def __init__(
            self, inputn, hidden, offset, shape=(90000, 4), inc_weights=None):
        self.inputn = inputn
        self.hidden = hidden
        self.shape = shape
        self.inc_weights = inc_weights
        self.offset = offset
        self.l_velocity = .5
        self.__outc_weights = None

    def _nguyen_widerow(self, weights):
        sc_factor = .7 * pow(self.hidden, 1.0/self.inputn)
        numerator = sc_factor * weights
        denominator = np.power(np.sum(np.squere(weights), axis=1), .5)
        return numerator / denominator

    def _init_weights(self):
        randomw = np.random.uniform(-0.5, 0.5, self.shape)
        slices = (randomw[i] for i in range(self.shape[0]))
        self.__weights = map(self._nguyen_widerow, slices)

    def outc_weights(self):
        if not self.__outc_weights:
            self.__outc_weights = self._init_weights()
        return self.__outc_weights

    def differentiate(self, input_):
        act = self.activation(input_)
        return act * (1 - act)

    def activation(self, input_):
        weighted = np.multiply(input_, self.outc_weights)
        return 1. / (1 + math.exp(-sum(weighted)))

    def learn(self, input_, correct):
        error = np.multiply((correct - input_), self.differentiate(input_))
        offset_corr = np.multily(self.l_velocity, error)
        weights_corr = np.multiply(offset_corr, input_)
        self.inc_weights = self.inc_weights + weights_corr
        self.offset -= offset_corr

        return error

    def perceive(self, input_):
        weighted = np.multiply(input_, self.outc_weights)
        return self.activation(self.offset + sum(weighted))


class InputNeuron(object):
    def __init__(self, element, inputn, hidden, shape=(90000, 4)):
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


class OutputNeuron(Neuron):
    def learn(self, input_, correct):
        error = super(OutputNeuron, self).learn(input_, correct)
        idx = 0
        while True:
            try:
                self.hidden[idx].learn(error[idx])
                idx += 1
            except IndexError:
                break


class Network(object):
    def __init__(self, init_data):
        self.layer_size = len(init_data)
        self.input_layer = map(self.create_input, init_data)
        self.hidden_layer = map(self.create_hidden, init_data)
        self.output_layer = map(self.create_output, init_data)

    def create_input(self, element):
        return InputNeuron(element, self.layer_size, self.layer_size)

    def create_hidden(self, idx):
        return Neuron(self.input_layer, self.layer_size, self.layer_size)

    def create_output(self, idx):
        return OutputNeuron(
            self.hidden_layer, self.layer_size, self.layer_size)

    def learn(self, root_path):
        raise NotImplementedError
        # for i in paths_in_root_dir:
            #input_ = (neuron.learn(file_path) for neuron in self.input_layer)
            #hidden = (neuron.learn(input_) for neuron in self.hidden_layer)
            #out = (neuron.learn(hidden) for neuron in self.output_layer)
            #if math.sqrt(pow(out, 2)) <= self.sqrt_lim:
            #    break

    def recognise(self, file_path):
        input_ = (neuron.perceive(file_path) for neuron in self.input_layer)
        hidden = (neuron.perceive(input_) for neuron in self.hidden_layer)
        return (neuron.perceive(hidden) for neuron in self.output_layer)
