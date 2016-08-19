import numpy as np


class Neuron(object):
    def __init__(self, ntype, inputn, hidden, shape=(90000, 4)):
        self.type = ntype
        self.inputn = inputn
        self.hidden = hidden
        self.shape = shape
        self.weights = self._init_weights()

    def nguyen_widerow(self, n, init_weights):
        sc_factor = .7 * pow(self.hidden, 1.0/self.inputn)
        numerator = sc_factor * init_weights[n]
        denominator = np.power(np.sum(np.squere(init_weights), axis=1), .5)
        return numerator / denominator

    def _init_weights(self):
        randomw = np.random.uniform(-0.5, 0.5, self.shape)
        #TODO: use nguyen_widerow
