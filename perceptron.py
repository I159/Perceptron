import cPickle
from bson.binary import Binary
import os
import string
import types

import numpy
from profilehooks import profile
from pymongo import MongoClient

from elements import Associative
from elements import ImageSize
from elements import Reaction
from elements import Sensor


class Neuron(object):
    def __init__(self, size, letter, threshold_coefficient=2.5):
        mongod = MongoClient()
        self.table = mongod.perceptron_db.weights
        self.letter = letter
        self.size = size
        self.shape = (size.X * size.Y, 4)
        self.weights = numpy.zeros(self.shape)
        self.threshold = self.shape[0] * threshold_coefficient
        self.bg_diff = None
        self.__weights = numpy.array(
            cPickle.loads(
                self.table.find_one({"letter": self.letter})['weights']),
            dtype=types.FloatType)

    @property
    def weights(self):
        if self.__weights is None:
            self.table.insert_one(
                {
                    "letter": self.letter,
                    "weights": Binary(cPickle.dumps(numpy.zeros(self.shape), protocol=2))
                })
            self.__weights = self.table.find_one({"letter": self.letter})
            self.__weights = cPickle.loads(self.__weights['weights'])
        return self.__weights

    @weights.setter
    def weights(self, weights):
        weights = Binary(cPickle.dumps(weights, protocol=2))
        self.table.update_one(
            {'letter': self.letter}, {"$set": {'weights': weights}})
        self.__weights = self.table.find_one({"letter": self.letter})
        self.__weights = cPickle.loads(self.__weights['weights'])
        return self.__weights

    def _decide(self, file_path):
        pixel_array = Sensor(file_path, self.size)
        self.bg_diff = Associative(pixel_array)
        return Reaction(self.threshold, self.weights, self.bg_diff)

    @profile
    def learn(self, file_path, correct_answer):
        positive = self._decide(file_path)

        if positive and correct_answer is False:
            weights = self.weights - self.bg_diff
        elif not positive and correct_answer is True:
            weights = self.weights + self.bg_diff
        else:
            return
        self.weights = weights

    @profile
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
            try:
                neuron.learn(next(true_imgs), True)
                neuron.learn(next(false_imgs), False)
            except StopIteration:
                break

    def recognize(self, root_path):
        for path in self._image_paths(root_path):
            for neuron in self.neurons.itervalues():
                result = neuron.recognize(path)
                if result:
                    yield result
