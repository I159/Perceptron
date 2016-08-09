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

    def learn(self, input_signal, decision):
        difference = self.get_bg_rel_diff(input_signal)
        result = self._recognize(difference)

        if result is True and decision is False:
            self.weights -= difference
        elif result is False and decision is True:
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


# TODO: Use false and true data items together to keep weights balanced
class Network(object):
    def __init__(self, image_size, letters):
        pix_num = image_size[0] * image_size[1]
        self.neurons = [Neuron(pix_num, i) for i in letters]
        self.image_size = image_size

    def _use_learning_data(self, true_path, false_path):
        true_paths = os.listdir(true_path)
        false_paths = os.listdir(false_path)
        true_file_paths = (os.path.join(true_path, i) for i in true_paths)
        false_file_paths = (os.path.join(false_path, i) for i in false_paths)
        file_paths = zip(true_file_paths, false_file_paths)
        for true, false in file_paths:
            try:
                true_img = Image.open(true)
                false_img = Image.open(false)
                # FIXME: gif images milformed after thumbnailing
                true_img.thumbnail(self.image_size, Image.ANTIALIAS)
                false_img.thumbnail(self.image_size, Image.ANTIALIAS)
                true_bordered = Image.new(
                    'RGBA', self.image_size, (255, 255, 255, 0))
                false_bordered = Image.new(
                    'RGBA', self.image_size, (255, 255, 255, 0))
                true_bordered.paste(
                   true_img,
                    (
                        (self.image_size[0] - true_img.size[0]) / 2,
                        (self.image_size[1] - true_img.size[1]) / 2)
                    )
                false_bordered.paste(
                    false_img,
                    (
                        (self.image_size[0] - false_img.size[0]) / 2,
                        (self.image_size[1] - false_img.size[1]) / 2)
                    )
                yield (numpy.array(true_bordered.getdata(), dtype=float, ndmin=2),
                       numpy.array(false_bordered.getdata(), dtype=float, ndmin=2))
            except IOError:
                continue

    def learn(self, true_path, false_path):
        print "Learning"
        for true, false in self._use_learning_data(true_path, false_path):
            for v in self.neurons:
                v.learn(true, True)
                v.learn(false, False)

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
