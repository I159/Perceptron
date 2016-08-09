import cProfile
import os
import string
import unittest

import numpy
from PIL import Image
import profilehooks


class Neuron(object):
    def __init__(self, size, letter):
        self.letter = letter
        self.size = float(size)
        self.weights = numpy.array([0] * size, dtype=float)
        self.threshold = self.size * 2.5 # / float(33)
        self.pixel_length = 4

    def learn(self, input_signal, letter):
        self.inc_letter = letter
        difference = self.get_bg_rel_diff(input_signal)
        result = self._recognize(difference)

        if result is True and letter != self.letter:
            self.weights -= difference
        elif result is False and letter == self.letter:
            self.weights += difference

    @profilehooks.profile
    def get_bg_rel_diff(self, input_signal):
        view_shape = [('', input_signal.dtype)]*input_signal.shape[1]
        view = input_signal.view(view_shape)
        unique_a = numpy.unique(view, return_counts=True)
        max_count = unique_a[0][unique_a[1].argmax(axis=0)].tolist()
        background = numpy.array(max_count, dtype=float)
        diff = numpy.absolute(numpy.subtract(input_signal, background)) / 256.0
        return numpy.mean(diff, axis=1)

    def _recognize(self, input_signal):
        mul_sum = sum(input_signal * self.weights)
        decision = bool(mul_sum >= self.threshold)
        return decision

    def recognize(self, input_signal, letter=None):
        self.inc_letter = letter
        difference = self.get_bg_rel_diff(input_signal)
        return self._recognize(difference)


class Network(object):
    def __init__(self, image_size, letters):
        pix_num = image_size[0] * image_size[1]
        self.neurons = [Neuron(pix_num, i) for i in letters]
        self.image_size = image_size

    def _use_learning_data(self, root_path):
        paths = os.listdir(root_path)
        file_paths = (os.path.join(root_path, i) for i in paths)
        for i in file_paths:
            try:
                img = Image.open(i)
                # FIXME: gif images milformed after thumbnailing
                img.thumbnail(self.image_size, Image.ANTIALIAS)
                bordered = Image.new(
                    'RGBA', self.image_size, (255, 255, 255, 0))
                bordered.paste(
                    img,
                    (
                        (self.image_size[0] - img.size[0]) / 2,
                        (self.image_size[1] - img.size[1]) / 2)
                    )
                yield numpy.array(bordered.getdata(), dtype=float, ndmin=2)
            except IOError:
                continue

    def learn(self, root_path, letter):
        print "Learning"
        for i in self._use_learning_data(root_path):
            for v in self.neurons:
                v.learn(i, letter)

    def recognize(self, root_path, letter=None):
        print "Recognition {}".format(letter)
        for pixel_array in self._use_learning_data(root_path):
            for neuron in self.neurons:
                result = neuron.recognize(pixel_array, letter)
                if result is True:
                    yield neuron.letter


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.network = Network((300, 300), string.ascii_lowercase)

    def test_learn(self):
        self.network.learn("/home/i159/Downloads/learning_data/a", 'a')
        self.network.learn("/home/i159/Downloads/learning_data/b", 'b')


if __name__ == '__main__':
    #pr = cProfile.Profile(subcalls=False)
    #pr.enable()

    network = Network((300, 300), string.ascii_lowercase)
    network.learn("/home/i159/Downloads/learning_data/a", 'a')
    network.learn("/home/i159/Downloads/learning_data/b", 'b')

    for i in network.recognize("/home/i159/Downloads/test_data/a", 'a'):
        print i
    for i in network.recognize("/home/i159/Downloads/test_data/w", 'b'):
        print i

    #pr.disable()
    #s = StringIO.StringIO()
    #sortby = 'cumulative'
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print s.getvalue()
