import collections
import cProfile
import os
import string

from PIL import Image
import profilehooks
import unittest


class Neuron(object):
    def __init__(self, size, letter):
        self.letter = letter
        self.size = size
        self.weights = [0] * size
        self.threshold = size / float(1000)
        self.pixel_length = 4

    def learn(self, input_signal, letter):
        print "Learning of letter -={}=- at the neuron -={}=- with.".format(
                letter, self.letter)
        difference = self.get_bg_rel_diff(input_signal)
        result = self.recognize(difference)

        if result == True and letter != self.letter:
            for i, j in enumerate(difference):
                self.weights[i] -= j
        elif result == False and letter == self.letter:
            for i, j in enumerate(difference):
                self.weights[i] += j

    def _diff_by_color(self, x):
        for idx in xrange(self.pixel_length):
            i = x[0][idx]
            j = x[1][idx]
            i, j = (i, j) if i < j else (j, i)
            if i == j == 0:
                yield 0
            elif i == 0:
                yield 1
            else:
                yield float(i)/float(j)

    def _percent_diff(self, x):
        return 1 - sum(self._diff_by_color(x)) / float(self.pixel_length)

    def get_bg_rel_diff(self, input_signal):
        counter = collections.Counter(input_signal)
        most_common = counter.most_common()
        background = most_common[0]

        result = []
        for i in input_signal:
            result.append(self._percent_diff((i, background[0])))
        return result

    def recognize(self, input_signal):
        multiplied = (i * j for i, j in zip(input_signal, self.weights))
        return sum(multiplied) >= self.threshold

class Network(object):
    def __init__(self, image_size, letters):
        # Use default quantity or calculate a quantity of neurons during
        # learning process
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
                yield bordered.getdata()
            except IOError:
                continue

    def learn(self, root_path, letter):
        for i in self._use_learning_data(root_path):
            for v in self.neurons:
                v.learn(i, letter)


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.network = Network((300, 300), string.ascii_lowercase)

    def test_learn(self):
        self.network.learn("/home/i159/Downloads/learning_data/a", 'a')
        self.network.learn("/home/i159/Downloads/learning_data/b", 'b')


if __name__ == '__main__':
    network = Network((300, 300), string.ascii_lowercase)
    network.learn("/home/i159/Downloads/learning_data/a", 'a')
