import os

from layers import InputLayer
from layers import HiddenLayer
from layers import OutputLayer
from neurons import InputNeuron
from neurons import HiddenNeuron
from neurons import OutputNeuron


class Network(object):
    def __init__(self, input_size, hidden_size, output_size):
        # TODO: Use layers registration
        self.output_layer = OutputLayer(
            OutputNeuron, input_size, hidden_size, output_size)
        self.hidden_layer = HiddenLayer(
            HiddenNeuron, input_size, hidden_size, output_size)
        self.input_layer = InputLayer(InputNeuron, input_size, hidden_size)

        self.input_layer.register_next_layer(self.hidden_layer)
        self.hidden_layer.register_previous_layer(self.input_layer)
        self.hidden_layer.register_next_layer(self.output_layer)
        self.output_layer.register_previous_layer(self.hidden_layer)

    def learn(self, root_path, correct):
        images = (os.path.join(root_path, i) for i in os.listdir(root_path))
        for neuron in self.input_layer:
            for image in images:
                neuron.perceive(image, correct)

    def recognise(self, file_path):
        input_ = (neuron.perceive(file_path) for neuron in self.input_layer)
        hidden = (neuron.perceive(input_) for neuron in self.hidden_layer)
        return (neuron.perceive(hidden) for neuron in self.output_layer)
