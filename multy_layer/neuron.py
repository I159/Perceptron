import numpy as np


class Neuron(object):
    def __init__(self, shape=(90000, 4)):
        self.shape = shape
        self.weights = self._init_weights()

    def _init_weights(self):
        return np.random.uniform(-0.5, 0.5, self.shape)
