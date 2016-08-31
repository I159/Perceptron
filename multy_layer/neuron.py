import math
import string
import unittest
import uuid

import mock
from PIL import Image

import numpy as np


class Layer(object):
    """Container type for neurons layer bulk operations."""
    def __init__(self, neuron_type, number, previous_layer=None, auto_id=True):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError


class OutputNeuron(object):
    def __init__(self, id_, hidden_layer, inputn, outpn, offset, l_velocity):
        self.id_ = id_
        self.outpn = outpn
        self.l_velocity = l_velocity
        self.hidden_n = len(hidden_layer)
        self.inputn = inputn
        self.hidden_layer = hidden_layer
        self.inc_weights = map(
            lambda x: x.outc_weights[self], hidden_layer)
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


class WeightsMixIn(object):
    def __init__(self, next_layer, inputn):
        self.next_layer = next_layer
        self.inputn = inputn
        self.__weights = None

    def _nguyen_widerow(self, widx):
        sc_factor = .7 * pow(len(self.next_layer), 1.0/self.inputn)
        numerator = np.multiply(self.__weights[widx], sc_factor)
        denominator = np.square(self.__weights[:widx+1])
        denominator = np.sum(denominator)
        denominator = math.sqrt(denominator)
        return numerator / denominator

    def _init_weights(self):
        self.__weights = np.random.uniform(-0.5, 0.5, len(self.next_layer))
        indexed_keys = enumerate(self.next_layer.iterkeys())
        return {k: self._nguyen_widerow(i) for i, k in indexed_keys}

    @property
    def weights(self):
        if not self.__weights:
            self.__weights = self._init_weights()
        return self.__weights


# TODO: use WeightsMixIn
class HiddenNeuron(object):

    def __init__(
            self, id_, input_layer, hidden_n, outc_n, offset):
        self.id_ = id_
        self.input_layer = input_layer
        self.inputn = len(input_layer)
        self.hidden = hidden_n
        self.outc_n = outc_n
        self.offset = offset
        self.l_velocity = .5
        self.output_errors = []

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
                            self.weights))
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


# TODO: use WeightsMixIn
class InputNeuron(object):
    def __init__(self, hidden_layer, inputn, shape=(90000, 4)):
        self.inputn = inputn
        self.hidden_layer = hidden_layer
        self.shape = shape
        self.image_size = (300, 300)
        self.__weights = None

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



class Network(object):
    def __init__(self, init_data, input_size, hidden_size, output_size):
        self.input_layer = Layer(InputNeuron, 28)
        self.hidden_layer = Layer(HiddenNeuron, 900, self.input_layer)
        self.output_layer = Layer(OutputNeuron, 28, self.hidden_layer)
        #self.layer_size = len(init_data)
        #self.hidden_size = hidden_size
        #self.output_size = output_size
        #self.input_size = input_size
        #self.hidden_ids = [uuid.uuid4() for i in xrange(hidden_size)]
        #self.input_layer = map(self.create_input, init_data)
        #self.hidden_offset = self.init_hidden_offset()
        #self.hidden_layer = map(self.create_hidden, init_data)
        #self.output_offset = self.init_output_offset()
        #self.output_layer = map(self.create_output, init_data)

    #def init_hidden_offset(self):
        #random = np.random.uniform(-0.5, 0.5, self.hidden_size)
        #return dict(zip(self.hidden_ids, random))

    #def init_output_offset(self):
        #random = np.random.uniform(-0.5, 0.5, self.output_size)
        #return dict(zip(self.hidden_layer, random))

    #def create_input(self, element):
        #return InputNeuron(element, self.layer_size, self.layer_size)

    #def create_hidden(self, id_):
        #return HiddenNeuron(id_, self.input_layer, self.hidden_size,
                            #self.output_size, self.hidden_offset)

    #def create_output(self, id_):
        #return OutputNeuron(id_, self.hidden_layer, self.input_size,
                            #self.output_size, self.output_offset, 0.5)

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


class TestWeights(unittest.TestCase):
    def test_input_weights(self):
        hidden_layer = mock.MagicMock()
        ids = [uuid.uuid4() for i in xrange(900)]
        hidden_layer.iterkeys.return_value = ids
        hidden_layer.__len__.return_value = len(ids)
        neuron = InputNeuron(hidden_layer, 28)
        self.assertEqual(len(neuron.weights), 900)
