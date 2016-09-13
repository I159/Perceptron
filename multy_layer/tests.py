import unittest
import uuid

import mock

from network import Network
from neurons import InputNeuron
from neurons import HiddenNeuron
from neurons import OutputNeuron


class TestWeights(unittest.TestCase):
    def test_input_weights(self):
        hidden_layer = mock.MagicMock()
        ids = [uuid.uuid4() for i in xrange(900)]
        hidden_layer.iterkeys.return_value = ids
        hidden_layer.__len__.return_value = len(ids)
        neuron = InputNeuron(hidden_layer, 28)
        self.assertEqual(len(neuron.weights), 900)

    def test_hidden_weights(self):
        output_layer = mock.MagicMock()
        ids = [uuid.uuid4() for i in xrange(900)]
        output_layer.iterkeys.return_value = ids
        output_layer.__len__.return_value = len(ids)
        neuron = HiddenNeuron('a', 28, 900, 28, mock.MagicMock())
        neuron.next_layer = mock.MagicMock()
        self.assertEqual(len(neuron.weights), 900)

    def test_output_weights(self):
        hidden_layer = mock.MagicMock()
        mock_iter = [mock.Mock(), mock.Mock(), mock.Mock()]
        for i in mock_iter:
            i.outc_weights.__getitem__ = mock.Mock(return_value=1)
        hidden_layer.__iter__ = mock.MagicMock(return_value=iter(mock_iter))
        neuron = OutputNeuron('a', hidden_layer, 28, 28,
            mock.MagicMock(), .5)
        self.assertEqual(len(neuron.inc_weights), 3)

    def test_init_network(self):
        network = Network(28, 900, 28)
