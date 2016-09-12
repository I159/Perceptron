import abc


class LayerNotRegistered(Exception):
    pass


class LayerAlreadyRegistered(Exception):
    pass


# TODO: Create different layers objects
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

    def register_layer(self, layer, previous=False, next_=False):
        if previous and not next_:
            for i in self.neurons:
                i.previous_layer = layer
        elif next_ and not previous:
            for i in self.neurons:
                i.next_layer = layer
        else:
            raise TypeError(
                "`previous` or `next_` layer type must be specified")

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


class OutputLayer(Layer):
    def __init__(self, neuron_type, number, shape=(90000, 4)):
        raise NotImplementedError()

    @staticmethod
    def _init_neuron():
        raise NotImplementedError()


class HiddenLayer(Layer):
    def __init__(self, neuron_type, number, shape=(90000, 4)):
        raise NotImplementedError()

    @staticmethod
    def _init_neuron():
        raise NotImplementedError()
