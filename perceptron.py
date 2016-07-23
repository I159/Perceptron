import os

from PIL import Image
import unittest


class Perceptron(object):
    def __init__(self, size):
        self.size = size
        self.weights = [[0]*size[0]]*size[1]

    def __repr__(self):
        return "Perceptron({}x{})".format(*self.size)

    def learn(self, input_signal):
        raise NotImplementedError

    def percive(self, input_signal):
        # get input signal in initial format (image)
        # split it to a matrix of pixels
        # associate it to the weights matrix
        raise NotImplementedError

    def _mul_signal_weight(self, input_signal):
        # Multiply pixel signals to the weight of an appropriate elements of
        # weight matrix
        raise NotImplementedError

    def _sum_signal_weight(self, input_signal):
        raise NotImplementedError

    def get_result(self):
        # Compare result with the threshold
        # Return boolean
        raise NotImplementedError


class Network(object):
    def __init__(self, quantity=None):
        # Use default quantity or calculate a quantity of neurons during
        # learning process
        self.quantity = quantity

    def open_image(self, image):
        raise NotImplementedError

    def use_learning_data(self, root_path):
        paths = os.listdir(root_path)
        file_paths = (os.path.join(root_path, i) for i in paths)
        # open images with PIL
        # Convert to a pixel matrix
        # Learn with the matrixes

    def learn(self, input_signal):
        # If no neurons presented - create and learn it
        # If every neurons are in initial state - learn first
        # else choose the most appropriate neuron
        # if there is no appropriate neuron - create new and learn (or learn
        # next one from initial state)
        raise NotImplementedError


class TestNetworkDefaultQuantity(unittest.TestCase):

    def setUp(self):
        self.network = Network(32) # Cyrillic alphabet

    def test_use_learning_data(self):
        self.network.use_learning_data("/home/i159/Downloads/learning_data")
