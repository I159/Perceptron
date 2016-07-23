class Perceptron(object):
    def __init__(self, size):
        self.size = size
        self.weights = [[0]*size[0]]*size[1]

    def __repr__(self):
        return "Perceptron({}x{})".format(*self.size)

    def learn(self, input_signal):
        raise NotImplementedError

    def percive(self, input_signal):
        # get input signal in initial format (image)
        # split it to a matrix of pixels
        # associate it to the weights matrix
        raise NotImplementedError

    def _mul_signal_weight(self, input_signal):
        # Multiply pixel signals to the weight of an appropriate elements of
        # weight matrix
        raise NotImplementedError

    def _sum_signal_weight(self, input_signal):
        raise NotImplementedError

    def get_result(self):
        # Compare result with the threshold
        # Return boolean
        raise NotImplementedError


class Network(object):
    def __init__(self, quantity=None):
        # Use default quantity or calculate a quantity of neurons during
        # learning process
        raise NotImplementedError

    def learn(self, input_signal):
        # If no neurons presented - create and learn it
        # If every neurons are in initial state - learn first
        # else choose the most appropriate neuron
        # if there is no appropriate neuron - create new and learn (or learn
        # next one from initial state)
        raise NotImplementedError


if __name__== "__main__":
    perceptron = Perceptron((100, 100))
    print perceptron
