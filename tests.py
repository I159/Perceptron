import unittest

from perceptron import Network


class TestRecognition(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.network = Network()
        a_true_path = "/home/i159/Dropbox/learning_data/a_true"
        a_false_path = "/home/i159/Dropbox/learning_data/a_false"
        b_true_path = "/home/i159/Dropbox/learning_data/b_true"
        b_false_path = "/home/i159/Dropbox/learning_data/b_false"

        self.network.learn(a_true_path, a_false_path, 'a')
        self.network.learn(b_true_path, b_false_path, 'b')

    def test_recognize_a(self):
        recognized = self.network.recognize("/home/i159/Dropbox/test_data/a")
        assert len([i for i in recognized if i == 'a']) >= 8

    def test_recognize_b(self):
        recognized = self.network.recognize("/home/i159/Dropbox/test_data/b")
        assert len([i for i in recognized if i == 'b']) >= 8
