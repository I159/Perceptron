import math

import numpy as np

from layers import LayerNotRegistered
from layers import LayerAlreadyRegistered


class WeightsMixIn(object):

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.__weights = None
        self.__next_layer = None

    def _nguyen_widerow(self, widx):
        sc_factor = .7 * pow(self.hidden_size, 1.0/self.input_size)
        numerator = np.multiply(self.weights[widx], sc_factor)
        denominator = np.square(self.weights[:widx+1])
        denominator = np.sum(denominator)
        denominator = math.sqrt(denominator)
        return numerator / denominator

    def _init_weights(self):
        try:
            self.__weights = np.random.uniform(-0.5, 0.5, len(self.next_layer))
            indexed_keys = enumerate(self.next_layer)
            return {k: self._nguyen_widerow(i) for i, k in indexed_keys}
        except LayerNotRegistered as e:
            raise LayerNotRegistered(
                "You can't initialize weights before next layer registered")

    @property
    def weights(self):
        if not self.__weights:
            self.__weights = self._init_weights()
        return self.__weights

    @property
    def next_layer(self):
        if self.__next_layer:
            return self.__next_layer
        raise LayerNotRegistered("Please register a next layer or ensure that"
                " the type of neuron requires a next layer neurons.")

    @next_layer.setter
    def next_layer(self, layer):
        if not self.__next_layer:
            self.__next_layer = layer
        else:
            raise LayerAlreadyRegistered()
