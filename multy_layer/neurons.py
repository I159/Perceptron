import math

import numpy as np
from PIL import Image

from multi_layer.layers import LayerNotRegistered
from multi_layer.layers import LayerAlreadyRegistered
from multi_layer.mixins import WeightsMixIn


class OutputNeuron(object):
    def __init__(self, id_,
            input_size, hidden_size, outp_size, offset, l_velocity):
        self.id_ = id_
        self.output_size = outp_size
        self.l_velocity = l_velocity
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.offset = offset
        self.__hidden_layer = None

    @property
    def inc_weights(self):
        self.__inc_weights = map(
            lambda x: x.outc_weights[self], self.hidden_layer)

    @property
    def hidden_layer(self):
        if self.__hidden_layer:
            return self.__hidden_layer
        raise LayerNotRegistered()

    @hidden_layer.setter
    def hidden_layer(self, layer):
        if not self.__hidden_layer:
            self.__hidden_layer = layer
        else:
            raise LayerAlreadyRegistered()

    def _nguyen_widerow(self, widx):
        sc_factor = .7 * pow(self.hidden_size, 1.0/self.input_size)
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


class HiddenNeuron(WeightsMixIn):

    def __init__(self, id_, input_size, hidden_size, output_size, offset):
        super(HiddenNeuron).__init__(input_size, hidden_size)
        self.id_ = id_
        self.output_size = output_size
        self.offset = offset
        self.l_velocity = .5
        self.output_errors = []
        self.__previous_layer = None

    @property
    def previous_layer(self):
        if self.__previous_layer:
            return self.__previous_layer
        raise LayerNotRegistered()

    @previous_layer.setter
    def previous_layer(self, layer):
        if not self.__previous_layer:
            self.__previous_layer = layer
        else:
            raise LayerAlreadyRegistered()

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
