class Neuron(object):
    def __init__(self, size):
        # default weights
        # default threshold
        # create signal matrix
        raise NotImplementedError

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
