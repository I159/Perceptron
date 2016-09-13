import abc
import collections
import uuid

from neurons import OutputNeuron


Offset = collections.namedtuple('Offset', ('income', 'outcome'))


class Layer(object):
    """Container type for neurons layer bulk operations."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        self.neurons = []
        raise NotImplementedError()

    @abc.abstractmethod
    def _init_neuron(self, input_size, hidden_size):
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
    def __init__(self, neuron_type, input_size, hidden_size):
        self.neuron_type = neuron_type
        neurons_factory = self._init_neuron(input_size, hidden_size)
        self.neurons = [neurons_factory() for i in xrange(input_size)]

    def _init_neuron(self, input_size, hidden_size):
        return lambda: self.neuron_type(input_size, hidden_size)


class OutputLayer(Layer):
    def __init__(self, neuron_type, input_size, hidden_size, output_size):
        self.neuron_type = neuron_type
        neurons_factory = self._init_neuron(
            input_size, hidden_size, output_size)
        self.offset = Offset(.5, .5)
        self.neurons = [neurons_factory() for i in xrange(output_size)]

    def _init_neuron(self, input_size, hidden_size, output_size):
        return lambda: self.neuron_type(uuid.uuid4(), input_size, hidden_size,
                                        output_size, self.offset, .5)

class HiddenLayer(Layer):
    # TODO: dry init method
    def __init__(self, neuron_type, input_size, hidden_size, output_size):
        self.neuron_type = neuron_type
        neurons_factory = self._init_neuron(
            input_size, hidden_size, output_size)
        self.offset = Offset(.5, .5)
        self.neurons = [neurons_factory() for i in xrange(hidden_size)]

    def _init_neuron(self, input_size, hidden_size, output_size):
        return lambda: self.neuron_type(uuid.uuid4(), input_size, hidden_size,
                                        output_size, self.offset)
