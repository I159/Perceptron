import abc
import collections
import uuid

from neurons import OutputNeuron


class Layer(object):
    """Container type for neurons layer bulk operations."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, neuron_type, number):
        self.neurons = []

    @staticmethod
    @abc.abstractmethod
    def _init_neuron(neuron_type):
        raise NotImplementedError()

    def register_previous_layer(self, layer):
        for i in self.neurons:
            i.previous_layer = layer

    def register_next_layer(self, layer):
        for i in self.neurons:
            i.next_layer = layer

    def __len__(self):
        return len(self.neurons)

    def __iter__(self):
        return iter(self.neurons)


class InputLayer(Layer):
    def __init__(self, neuron_type, number, shape=(90000, 4)):
        raise NotImplementedError()

    @staticmethod
    def _init_neuron():
        raise NotImplementedError()


Offset = collections.namedtuple('Offset', ('income', 'outcome'))


class OutputLayer(Layer):
    def __init__(self, neuron_type, input_size, hidden_size, output_size):
        neurons_factory = self._init_neuron(
            input_size, hidden_size, output_size)
        self.offset = Offset(.5, .5)
        self.neurons = [neurons_factory() for i in xrange(output_size)]

    def _init_neuron(self, input_size, hidden_size, output_size):
        return lambda: OutputNeuron(uuid.uuid4(), input_size, hidden_size,
                                    output_size, self.offset, .5)


class HiddenLayer(Layer):
    def __init__(self, neuron_type, number, shape=(90000, 4)):
        raise NotImplementedError()

    @staticmethod
    def _init_neuron():
        raise NotImplementedError()
