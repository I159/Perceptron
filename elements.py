import collections
import types

import numpy
from PIL import Image


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

    def __new__(cls, buff):
        background = cls._get_background(numpy.array(buff))
        diff = numpy.subtract(buff, background)
        abs_diff = numpy.absolute(diff) / 256.0
        return abs_diff

    @classmethod
    def _get_background(self, input_signal):
        view_shape = [('', input_signal.dtype)]*input_signal.shape[1]
        view = input_signal.view(view_shape)
        unique_a = numpy.unique(view, return_counts=True)
        max_count = unique_a[0][unique_a[1].argmax(axis=0)].tolist()
        return numpy.array(max_count, dtype=float)


class Reaction(object):
    """R element"""
    def __init__(self, threshold, weights, diff):
        self.__bool__ = bool(numpy.sum(diff * weights) >= threshold)

    def __repr__(self):
        return "Reaction({})".format(self.__bool__)

    def __nonzero__(self):
        return self.__bool__
