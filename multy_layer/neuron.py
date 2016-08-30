import math
from PIL import Image
import unittest

import numpy as np


class OutputNeuron(object):
    def __init__(self, hidden_layer, inputn, outpn, offset, l_velocity):
        self.outpn = outpn
        self.l_velocity = l_velocity
        self.hidden_n = len(hidden_layer)
        self.inputn = inputn
        self.hidden_layer = hidden_layer
        self.inc_weights = map(lambda x: x.outc_weights[self], hidden_layer)
        self.offset = offset

    def _nguyen_widerow(self, widx):
        sc_factor = .7 * pow(self.hidden_n, 1.0/self.inputn)
        numerator = np.multiply(self.inc_weights[widx], sc_factor)
        denominator = np.square(self.inc_weights[:widx+1])
        denominator = np.sum(denominator)
        denominator = math.sqrt(denominator)
        return numerator / denominator

    def differentiate(self, input_):
        act = self.activation(input_)
        return act * (1 - act)

    def activation(self, input_):
        weighted = np.multiply(input_, self.inc_weights)
        return 1. / (1 + math.exp(-sum(weighted)))

    def learn(self, input_, correct):
        error = np.multiply((correct - input_), self.differentiate(input_))
        offset_corr = np.multily(self.l_velocity, error)
        weights_corr = np.multiply(offset_corr, input_)

        for i in self.hidden_layer:
            i.learn(error)

        self._change_weights(weights_corr)
        self._change_offset(offset_corr)

    def _change_weights(self, correction):
        for neuron in self.hidden_layer:
            neuron.outc_weights[self] += correction

    def _change_offset(self, correction):
        self.offset[self] += correction


class HiddenNeuron(object):

    def __init__(
            self, input_layer, hidden_n, outc_n, offset):
        self.input_layer = input_layer
        self.inputn = len(input_layer)
        self.hidden = hidden_n
        self.outc_n = outc_n
        self.offset = offset
        self.l_velocity = .5
        self.__outc_weights = None
        self.output_errors = []

    def _nguyen_widerow(self, widx):
        sc_factor = .7 * pow(self.hidden, 1.0/self.inputn)
        numerator = np.multiply(self.__outc_weights[widx], sc_factor)
        denominator = np.square(self.__outc_weights[:widx+1])
        denominator = np.sum(denominator)
        denominator = math.sqrt(denominator)
        return numerator / denominator

    def _init_weights(self):
        self.__outc_weights = np.random.uniform(-0.5, 0.5, self.outc_n)
        return map(self._nguyen_widerow, xrange(self.outc_n))

    @property
    def outc_weights(self):
        if not self.__outc_weights:
            self.__outc_weights = self._init_weights()
        return self.__outc_weights

    def differentiate(self, input_):
        act = self.activation(input_)
        return act * (1 - act)

    def activation(self, weighted_sum):
        return 1. / (1 + math.exp(-weighted_sum))

    def learn(self, error):
        self.output_errors.append(error)
        if len(self.output_errors) == self.outc_n:
            self.output_errors = np.array(self.output_errors)
            weighted_sum = sum(
                np.multiply(self.output_errors,
                            self.outc_weights))
            activation = self.activation(sum(weighted_sum))
            hidden_error = activation * weighted_sum
            offset_corr = np.multiply(self.l_velocity, hidden_error)
            weights_correction = np.multiply(self.output_errors, offset_corr)

            self._change_weights(weights_correction)
            self._change_offset(offset_corr)

    def _change_offset(self, correction):
        self.offset[self] += correction

    def _change_weights(self, correction):
        for neuron in self.input_layer:
            neuron.outc_weights[self] += correction

    def perceive(self, input_):
        weighted = np.multiply(input_, self.outc_weights)
        return self.activation(self.offset + sum(weighted))


class InputNeuron(object):
    def __init__(self, inputn, hidden, outputn, shape=(90000, 4)):
        self.inputn = inputn
        self.hidden = hidden
        self.outputn = outputn
        self.shape = shape
        self.image_size = (300, 300)
        self.__outc_weights = None

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

    def _nguyen_widerow(self, widx):
        sc_factor = .7 * pow(self.hidden, 1.0/self.inputn)
        numerator = np.multiply(self.__outc_weights[widx], sc_factor)
        denominator = np.square(self.__outc_weights[:widx+1])
        denominator = np.sum(denominator)
        denominator = math.sqrt(denominator)
        return numerator / denominator

    def _init_weights(self):
        self.__outc_weights = np.random.uniform(-0.5, 0.5, self.hidden)
        return map(self._nguyen_widerow, xrange(self.outputn))

    @property
    def outc_weights(self):
        if not self.__outc_weights:
            self.__outc_weights = self._init_weights()
        return self.__outc_weights


class Network(object):
    def __init__(self, init_data, input_size, hidden_size, output_size):
        self.layer_size = len(init_data)
        self.input_layer = map(self.create_input, init_data)
        self.hidden_layer = map(self.create_hidden, init_data)
        self.output_layer = map(self.create_output, init_data)
        self.hidden_offset = self.init_hidden_offset()
        self.output_offset = self.init_output_offset()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size

    def init_hidden_offset(self):
        random = np.random.uniform(-0.5, 0.5, len(self.hidden_layer))
        return dict(zip(self.hidden_layer, random))

    def init_output_offset(self):
        random = np.random.uniform(-0.5, 0.5, len(self.output_layer))
        return dict(zip(self.hidden_layer, random))

    def create_input(self, element):
        return InputNeuron(element, self.layer_size, self.layer_size)

    def create_hidden(self, idx):
        return HiddenNeuron(self.input_layer, self.hidden_size,
                            self.output_size, self.hidden_offset)

    def create_output(self, idx):
        return OutputNeuron(self.hidden_layer, self.input_size,
                            self.output_size, self.output_offset, 0.5)

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


class TestInitWeights(unittest.TestCase):
    def test_init_input_neuron(self):
        unittest.skip("Not implemented")

    def test_init_hidden_neuron(self):
        unittest.skip("Not implemented")

    def test_init_output_neuron(self):
        unittest.skip("Not implemented")

    def test_init_network(self):
        unittest.skip("Not implemented")
