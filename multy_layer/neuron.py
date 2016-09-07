import math
import string
import unittest
import uuid

import mock
from PIL import Image

import numpy as np


class OutputNeuron(object):
    def __init__(self, id_, hidden_layer, inputn, outpn, offset, l_velocity):
        self.id_ = id_
        self.outpn = outpn
        self.l_velocity = l_velocity
        self.hidden_n = len(hidden_layer)
        self.inputn = inputn
        self.hidden_layer = hidden_layer
        self.inc_weights = map(
            # TODO: iterable mock
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
    def __init__(self, next_layer_ids, input_size, hidden_size):
        self.next_layer_ids = next_layer_ids
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.__weights = None

    def _nguyen_widerow(self, widx):
        sc_factor = .7 * pow(self.hidden_size, 1.0/self.input_size)
        numerator = np.multiply(self.__weights[widx], sc_factor)
        denominator = np.square(self.__weights[:widx+1])
        denominator = np.sum(denominator)
        denominator = math.sqrt(denominator)
        return numerator / denominator

    def _init_weights(self):
        self.__weights = np.random.uniform(-0.5, 0.5, len(self.next_layer_ids))
        indexed_keys = enumerate(self.next_layer_ids)
        return {k: self._nguyen_widerow(i) for i, k in indexed_keys}

    @property
    def weights(self):
        if not self.__weights:
            self.__weights = self._init_weights()
        return self.__weights


class HiddenNeuron(WeightsMixIn):

    def __init__(self, id_, previous_layer, next_layer, input_size,
            hidden_size, output_size, offset):
        super(HiddenNeuron, self).__init__(
            next_layer.iterkeys(), input_size, hidden_size)
        self.id_ = id_
        self.previous_layer = previous_layer
        self.output_size = output_size
        self.offset = offset
        self.l_velocity = .5
        self.output_errors = []

    def differentiate(self, previous_):
        act = self.activation(previous_)
        return act * (1 - act)

    def activation(self, weighted_sum):
        return 1. / (1 + math.exp(-weighted_sum))

    def learn(self, error):
        self.output_errors.append(error)
        if len(self.output_errors) == self.output_size:
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
        for neuron in self.previous_layer:
            neuron.weights[self] += correction

    def perceive(self, previous_):
        weighted = np.multiply(previous_, self.weights)
        return self.activation(self.offset + sum(weighted))


class InputNeuron(WeightsMixIn):
    def __init__(self, hidden_layer, input_size, shape=(90000, 4)):
        super(InputNeuron, self).__init__(
            hidden_layer.iterkeys(), input_size, len(hidden_layer))
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


class Network(object):
    def __init__(self, input_size, hidden_size, output_size):
        # FIXME: create a pattern to initialize objects dependent from each
        # other.
        self.output_layer = Layer(OutputNeuron, 28, self.hidden_layer)
        self.hidden_layer = Layer(HiddenNeuron, hidden_size, self.output_layer,
                input_size)
        self.input_layer = Layer(InputNeuron, input_size,
                input_size, next_layer=self.hidden_layer)

    def learn(self, root_path):
        raise NotImplementedError

    def recognise(self, file_path):
        input_ = (neuron.perceive(file_path) for neuron in self.input_layer)
        hidden = (neuron.perceive(input_) for neuron in self.hidden_layer)
        return (neuron.perceive(hidden) for neuron in self.output_layer)


class Layer(object):
    """Container type for neurons layer bulk operations."""
    def __init__(self, neuron_type, number, input_size, next_layer=None,
             previous_layer=None, shape=(90000, 4)):
        neuron_factory = self._init_input_neuron(
            neuron_type, next_layer, input_size, shape)
        self._neurons = map(neuron_factory, xrange(number))

    @staticmethod
    def _init_input_neuron(neuron_type, next_layer, input_size, shape):
        return lambda x: neuron_type(next_layer, input_size, shape)

    def create_neurons(self):
        # Create neurons using ids of different neurons layers.
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        return iter(self._neurons)

    def __next__(self):
        raise NotImplementedError


class TestWeights(unittest.TestCase):
    def test_input_weights(self):
        hidden_layer = mock.MagicMock()
        ids = [uuid.uuid4() for i in xrange(900)]
        hidden_layer.iterkeys.return_value = ids
        hidden_layer.__len__.return_value = len(ids)
        neuron = InputNeuron(hidden_layer, 28)
        self.assertEqual(len(neuron.weights), 900)

    def test_hidden_weights(self):
        output_layer = mock.MagicMock()
        ids = [uuid.uuid4() for i in xrange(900)]
        output_layer.iterkeys.return_value = ids
        output_layer.__len__.return_value = len(ids)
        neuron = HiddenNeuron('a', mock.MagicMock(), output_layer, 28, 900, 28,
            mock.MagicMock())
        self.assertEqual(len(neuron.weights), 900)

    def test_output_weights(self):
        hidden_layer = mock.MagicMock()
        mock_iter = [mock.Mock(), mock.Mock(), mock.Mock()]
        for i in mock_iter:
            i.outc_weights.__getitem__ = mock.Mock(return_value=1)
        hidden_layer.__iter__ = mock.MagicMock(return_value=iter(mock_iter))
        neuron = OutputNeuron('a', hidden_layer, 28, 28,
            mock.MagicMock(), .5)
        self.assertEqual(len(neuron.inc_weights), 3)

    def test_init_network(self):
        network = Network(28, 900, 28)
