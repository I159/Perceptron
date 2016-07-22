import collections


class Sensor(object):
    threshold = 0

    def __init__(self, weight=0):
        self.weight = weight

    def perceive(self, incoming_sig):
        # TODO: Logic of a signal processing
        return int(incoming_sig > self.weight)

    def learn(self, incoming_sig):
        self.weight += self.perceive(incoming_sig)

    def make_decision(self, incoming_sig):
        # TODO: calculate a coefficient and return a decision
        raise NotImplementedError


Size = collections.namedtuple("Size", ('X', 'Y'))


class Perceptron(object):
    def __init__(self, size):
        self.sensors = [[Sensor() for v in size.Y] for i in size.X]
        self.learn_mode = True

    def perceive(self, input_signal):
        # TODO: proceed all the image
        # Pass individual pixels to a Sensors instances
        raise NotImplementedError

    def toggle_mode(self):
        self.learn_mode = not self.learn_mode
