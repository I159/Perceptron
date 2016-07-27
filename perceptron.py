import collections
import cProfile
import os
import string
import pstats, StringIO

import numpy
from PIL import Image
import unittest


class Neuron(object):
    def __init__(self, size, letter):
        self.letter = letter
        self.size = size
        self.weights = numpy.array([0] * size, dtype=float)
        self.threshold = size / float(1000)
        self.pixel_length = 4

    def learn(self, input_signal, letter):
        print "Learning of letter -={}=- at the neuron -={}=- with.".format(
                letter, self.letter)
        difference = self.get_bg_rel_diff(input_signal)
        result = self.recognize(difference)

        if result is True and letter != self.letter:
            self.weights -= difference
        elif result is False and letter == self.letter:
            self.weights += difference

    @staticmethod
    def diff_by_color(item, background):
        # TODO: optimize!
        if not False in (item == background):
            gt = numpy.maximum(item, background, dtype=float)
            lt = numpy.minimum(item, background, dtype=float)
            res = numpy.nan_to_num(lt/gt)
            return numpy.sum(res) / 4
        return 0.

    def get_bg_rel_diff(self, input_signal):
        view_shape = [('', input_signal.dtype)]*input_signal.shape[1]
        view = input_signal.view(view_shape)
        unique_a = numpy.unique(view, return_counts=True)
        max_count = unique_a[0][unique_a[1].argmax(axis=0)].tolist()
        background = numpy.array(max_count, dtype=float)
        return numpy.apply_along_axis(self.diff_by_color, axis=1,
                arr=input_signal, background=background)

    def recognize(self, input_signal):
        mul_sum = sum(input_signal * self.weights)
        return mul_sum >= self.threshold

class Network(object):
    def __init__(self, image_size, letters):
        pix_num = image_size[0] * image_size[1]
        self.neurons = [Neuron(pix_num, i) for i in letters]
        self.image_size = image_size

    def _use_learning_data(self, root_path):
        paths = os.listdir(root_path)[:1]
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
        for i in self._use_learning_data(root_path):
            for v in self.neurons[:2]:
                v.learn(i, letter)


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.network = Network((300, 300), string.ascii_lowercase)

    def test_learn(self):
        self.network.learn("/home/i159/Downloads/learning_data/a", 'a')
        self.network.learn("/home/i159/Downloads/learning_data/b", 'b')


if __name__ == '__main__':
    pr = cProfile.Profile(subcalls=False)
    pr.enable()

    network = Network((300, 300), string.ascii_lowercase)
    network.learn("/home/i159/Downloads/learning_data/a", 'a')

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
