import collections
import cProfile
import os
import string
import types
import unittest

import numpy
from PIL import Image
import profilehooks


ImageSize = collections.namedtuple('ImageSize', ('X', 'Y'))


class Sensor(numpy.ndarray):
    """`S` elements wrapped to a single object"""
    def __new__(cls, image_file_path, size):
        cls.size = size
        cls.image_size = ImageSize(*size)
        return numpy.array(
            cls._perceive(image_file_path),
            dtype=types.FloatType,
            ndmin=2)

    @classmethod
    def _perceive(cls, file_path):
        image = Image.open(file_path)
        standardized_image = cls._standardize_size(image)
        return standardized_image.getdata()

    @classmethod
    def _standardize_size(cls, image):
        image.thumbnail(cls.image_size, Image.ANTIALIAS)
        bordered = Image.new('RGBA', cls.size, (255, 255, 255, 0))
        borders_size = (
            (cls.image_size[0] - image.size[0]) / 2,
            (cls.image_size[1] - image.size[1]) / 2)
        bordered.paste(image, borders_size)
        return bordered


class Associative(numpy.ndarray):
    """`A` elements wrapped to a single object"""

    def __array_finalize__(self, obj):
        background = self._get_background(self)
        diff = numpy.subtract(self, background)
        abs_diff = numpy.absolute(diff) / 256.0
        return numpy.mean(abs_diff, axis=1)

    def _get_background(self, input_signal):
        view_shape = [('', input_signal.dtype)]*input_signal.shape[1]
        view = input_signal.view(view_shape)
        unique_a = numpy.unique(view, return_counts=True)
        max_count = unique_a[0][unique_a[1].argmax(axis=0)].tolist()
        return numpy.array(max_count, dtype=float)


class Reaction(object):
    def __init__(self, threshold, weights, diff):
        self.diff = sum(diff * weights)
        self.threshold = threshold
        self.__bool = None

    def __bool__(self):
        return self.__nonzero__()

    def __nonzero__(self):
        if not self.__bool:
            self.__bool = self.diff >= self.threshold
        return self.__bool


class Neuron(object):
    def __init__(self, size, letter, threshold_coefficient=2.5):
        self.letter = letter
        self.size = size
        self.flat_size = size.X * size.Y
        self.weights = numpy.zeros(self.flat_size)
        self.threshold = self.flat_size * threshold_coefficient
        self.bg_diff = None

    def _decide(self, file_path):
        pixel_array = Sensor(file_path, self.size)
        self.bg_diff = Associative(shape=(4, self.flat_size), buffer=pixel_array)
        return Reaction(self.threshold, self.weights, self.bg_diff)

    def learn(self, file_path, correct_answer):
        decision = self._decide(file_path)

        if decision is True and correct_answer is False:
            self.weights -= self.bg_diff
        elif decision is False and correct_answer is True:
            self.weights += self.bg_diff

    def recognize(self, file_path):
        decision = self._decide(file_path)
        return (decision and self.letter) or False


class Network(object):

    def __init__(self, img_size=(300, 300)):
        self.img_size = ImageSize(*img_size)
        self.neurons = {
            i: Neuron(self.img_size, i) for i in string.ascii_lowercase}

    @staticmethod
    def _image_paths(path):
        return (os.path.join(path, i) for i in os.listdir(path))

    def learn(self, true_path, false_path, letter):
        neuron = self.neurons[letter]
        true_imgs = self._image_paths(true_path)
        false_imgs = self._image_paths(false_path)

        while True:
            neuron.learn(next(true_imgs), True)
            neuron.learn(next(false_imgs), False)

    def recognize(self, root_path):
        for path in self._image_paths(root_path):
            for neuron in self.neurons:
                result = neuron.recognize(path)
                if result:
                    yield result


class TestRecognition(unittest.TestCase):
    def setUp(self):
        self.network = Network()
        a_true_path = "/home/i159/Dropbox/learning_data/a_true"
        a_false_path = "/home/i159/Dropbox/learning_data/a_false"
        b_true_path = "/home/i159/Dropbox/learning_data/b_true"
        b_false_path = "/home/i159/Dropbox/learning_data/b_false"

        self.network.learn(a_true_path, a_false_path, 'a')
        self.network.learn(b_true_path, b_false_path, 'b')

    def test_recognize_a(self):
        recognized = self.network.recognize("/home/i159/Dropbox/test_data/a")
        assert len([i for i in recognized if i == 'a']) >= 8

    def test_recognize_b(self):
        recognized = self.network.recognize("/home/i159/Dropbox/test_data/b")
        assert len([i for i in recognized if i == 'b']) >= 8
