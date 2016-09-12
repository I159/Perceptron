from multi_layer.layers import InputLayer
from multi_layer.layers import HiddenLayer
from multi_layer.layers import OutputLayer
from multi_layer.neurons import InputNeuron
from multi_kayer.neurons import HiddenNeuron
from multi_layer.neurons import OutputNeuron


class Network(object):
    def __init__(self, input_size, hidden_size, output_size):
        # TODO: Use layers registration
        self.output_layer = OutputLayer(OutputNeuron, input_size, output_size)
        self.hidden_layer = HiddenLayer(HiddenNeuron, hidden_size, input_size)
        self.input_layer = InputLayer(InputNeuron, input_size, input_size)

    def learn(self, root_path):
        raise NotImplementedError

    def recognise(self, file_path):
        input_ = (neuron.perceive(file_path) for neuron in self.input_layer)
        hidden = (neuron.perceive(input_) for neuron in self.hidden_layer)
        return (neuron.perceive(hidden) for neuron in self.output_layer)
