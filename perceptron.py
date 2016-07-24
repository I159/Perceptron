import collections
import os

from PIL import Image
import unittest


class Neuron(object):
    def __init__(self, size):
        self.size = size
        self.weights = [0] * size
        self.threshold = size / float(1000)

    def is_init(self):
        return len(set(self.weights)) == 1

    def __repr__(self):
        return "Perceptron({}x{})".format(*self.size)

    def learn(self, input_signal): #, correct_answer):
        difference = self.get_bg_rel_diff(input_signal)
        result = self.recognize(difference)
        #if result == True and resut != correct_answer:
            #for i, j in zip(self.weights:

        #    do nothing
        # Else if result is incorrect:
        #    and true:
        #        deduct input signal values from the weights.
        #    else:
        #        add input signal values to the weights.

    def get_bg_rel_diff(self, input_signal):
        counter = collections.Counter(input_signal)
        most_common = counter.most_common()
        background = most_common[0]

        zip_with_bg = lambda x: zip(x, background[0])[:3]
        get_diff = lambda x: 1 - (sum(float(i)/float(j) for i, j in x) / 3.0)

        return map(get_diff, map(zip_with_bg, input_signal))

    def recognize(self, input_signal):
        multiplied = (i * j for i, j in zip(input_signal, self.weights))
        return sum(multiplied) >= self.threshold


class Network(object):
    def __init__(self, image_size, quantity=None):
        # Use default quantity or calculate a quantity of neurons during
        # learning process
        pix_num = image_size[0] * image_size[1]
        self.neurons = [Neuron(pix_num) for i in xrange(quantity)]
        self.image_size = image_size

    def _use_learning_data(self, root_path):
        paths = os.listdir(root_path)
        file_paths = (os.path.join(root_path, i) for i in paths)
        for i in file_paths:
            img = Image.open(i)
            # FIXME: gif images milformed after thumbnailing
            img.thumbnail(self.image_size, Image.ANTIALIAS)
            bordered = Image.new('RGBA', self.image_size, (255, 255, 255, 0))
            bordered.paste(
                    img,
                    (
                        (self.image_size[0] - img.size[0]) / 2,
                        (self.image_size[1] - img.size[1]) / 2)
                    )
            yield bordered.getdata()

    def learn(self, root_path):
        for i in self._use_learning_data(root_path):
            # TODO: learn each neuron different letters
            # Pass correct answer to a neuron
            for v in self.neurons:
                if v.is_init():
                    v.learn(i)
                    break
            # If no neurons presented - create and learn it
            # If every neurons are in initial state - learn first
            # else choose the most appropriate neuron
            # if there is no appropriate neuron - create new and learn (or learn
            # next one from initial state)


class TestNetworkDefaultQuantity(unittest.TestCase):

    def setUp(self):
        self.network = Network((300, 300), 32) # Cyrillic alphabet

    def test_learn(self):
        self.network.learn("/home/i159/Downloads/learning_data")
