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


class Sensor(numpy.array):
    """`S` elements wrapped to a single object"""
    image_size = (300, 300)

    def __init__(self, image_file_path):
        self.image_size = ImageSize(*self.image_size)
        self.ndmin = 2
        self.dtype = types.FloatType
        super(Sensor, self).__init__(self._perceive(image_file_path))

    def _perceive(self, file_path):
        image = Image.open(file_path)
        standardized_image = self._standardize_size(image)
        return numpy.array(standardized_image.get_data(), dtype=float, ndmin=2)

    def _standardize_size(self, image):
        image.thumbnail(self.image_size, Image.ANTIALIAS)
        bordered = Image.new('RGBA', self.image_size, (255, 255, 255, 0))
        borders_size = (
            (self.image_size[0] - image.size[0]) / 2,
            (self.image_size[1] - image.size[1]) / 2)
        bordered.paste(image, borders_size)
        return bordered


class Associative(numpy.array):
    """`A` elements wrapped to a single object"""

    def __array_finalize__(self, obj):
        background = self._get_background(obj)
        diff = numpy.subtract(obj, background)
        abs_diff = numpy.absolute(diff) / 256.0
        return numpy.mean(abs_diff, axis=1)

    def _get_background(self, input_signal):
        view_shape = [('', input_signal.dtype)]*input_signal.shape[1]
        view = input_signal.view(view_shape)
        unique_a = numpy.unique(view, return_counts=True)
        max_count = unique_a[0][unique_a[1].argmax(axis=0)].tolist()
        return numpy.array(max_count, dtype=float)


class Reaction(object):
    def __init__(self, threshold, diff):
        self.diff = sum(diff)
        self.threshold = threshold
        self.__bool = None

    def __bool__(self):
        return self.__nonzero__()

    def __nonzero__(self):
        if not self.__bool:
            self.__bool = self.diff >= self.threshold
        return self.__bool


class Neuron(object):
    def __init__(self, size, letter):
        self.letter = letter
        self.size = float(size)
        self.weights = numpy.array([0] * size, dtype=float)
        self.threshold = self.size * 2.5
        self.bg_diff = None

    def _decide(self, file_path):
        pixel_array = Sensor(file_path)
        self.bg_diff = Associative(pixel_array)
        return Reaction(self.threshold, self.bg_diff)

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
    pass


#class Neuron(object):
    #def __init__(self, size, letter):
        #self.letter = letter
        #self.size = float(size)
        #self.weights = numpy.array([0] * size, dtype=float)
        #self.threshold = self.size * 2.5 # / float(33)
        #self.pixel_length = 4

    #def learn(self, input_signal, correct_answer):
        #difference = self.get_bg_rel_diff(input_signal)
        #result = self._recognize(difference)

        #if result is True and correct_answer is False:
            #self.weights -= difference
        #elif result is False and correct_answer is True:
            #self.weights += difference

    #@profilehooks.profile
    #def get_bg_rel_diff(self, input_signal):
        #view_shape = [('', input_signal.dtype)]*input_signal.shape[1]
        #view = input_signal.view(view_shape)
        #unique_a = numpy.unique(view, return_counts=True)
        #max_count = unique_a[0][unique_a[1].argmax(axis=0)].tolist()
        #background = numpy.array(max_count, dtype=float)
        #diff = numpy.absolute(numpy.subtract(input_signal, background)) / 256.0
        #return numpy.mean(diff, axis=1)

    #def _recognize(self, input_signal):
        #mul_sum = sum(input_signal * self.weights)
        #decision = bool(mul_sum >= self.threshold)
        #return decision

    #def recognize(self, input_signal, letter=None):
        #self.inc_letter = letter
        #difference = self.get_bg_rel_diff(input_signal)
        #return self._recognize(difference)


## TODO: Use false and true data items together to keep weights balanced
#class Network(object):
    #def __init__(self, image_size, letters):
        #pix_num = image_size[0] * image_size[1]
        #self.neurons = {i: Neuron(pix_num, i) for i in letters}
        #self.image_size = image_size

    #def _get_pixel_array(self, path):
        #file_paths = (os.path.join(path, i) for i in os.listdir(path))
        #for pth in file_paths:
            #try:
                #img = Image.open(pth)
                #img.thumbnail(self.image_size, Image.ANTIALIAS)
                #bordered = Image.new(
                    #'RGBA', self.image_size, (255, 255, 255, 0))
                #bordered.paste(
                    #img,
                    #(
                        #(self.image_size[0] - img.size[0]) / 2,
                        #(self.image_size[1] - img.size[1]) / 2)
                    #)
                #yield numpy.array(bordered.getdata(), dtype=float, ndmin=2)
            #except IOError as err_msg:
                #print err_msg
                #continue

    #def _use_learning_data(self, true_path, false_path):
        #true_pxls = self._get_pixel_array(true_path)
        #false_pxls = self._get_pixel_array(false_path)
        #while True:
            #yield next(true_pxls), next(false_pxls)

    #def learn(self, true_path, false_path, letter):
        #print "Learning"
        #neuron = self.neurons[letter]
        #for true, false in self._use_learning_data(true_path, false_path):
            #neuron.learn(true, True)
            #neuron.learn(false, False)

    #def recognize(self, root_path, letter=None):
        #print "Recognition {}".format(letter)
        #for pixel_array in self._get_pixel_array(root_path):
            #for neuron in self.neurons.itervalues():
                #result = neuron.recognize(pixel_array, letter)
                #if result is True:
                    #yield neuron.letter


#class TestNetwork(unittest.TestCase):

    #def setUp(self):
        #self.network = Network((300, 300), string.ascii_lowercase)

    #def test_learn(self):
        #self.network.learn("/home/i159/Downloads/learning_data/a", 'a')
        #self.network.learn("/home/i159/Downloads/learning_data/b", 'b')


if __name__ == '__main__':
    #pr = cProfile.Profile(subcalls=False)
    #pr.enable()

    network = Network((300, 300), string.ascii_lowercase)
    a_true_path = "/home/i159/Dropbox/learning_data/a_true"
    a_false_path = "/home/i159/Dropbox/learning_data/a_false"
    b_true_path = "/home/i159/Dropbox/learning_data/b_true"
    b_false_path = "/home/i159/Dropbox/learning_data/b_false"

    network.learn(a_true_path, a_false_path, 'a')
    network.learn(b_true_path, b_false_path, 'b')

    for i in network.recognize("/home/i159/Dropbox/test_data/a", 'a'):
        print i
    for i in network.recognize("/home/i159/Dropbox/test_data/b", 'b'):
        print i

    #pr.disable()
    #s = StringIO.StringIO()
    #sortby = 'cumulative'
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print s.getvalue()
