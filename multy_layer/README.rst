* Activation function f(x) = 1/(1+e**x)

ALGORITHM
=========

0. Initialization of weights (weight of all the connections are initialized with
  small random values).

1. Iteratively repeat steps 2 - 9 until the termination condition of
  the algorithm is not reached.

2. For each pair: a data, a target value; perform steps 3 - 8.

3. Each input neuron sends the handled signal to all the neurons in the next
   hidden layer.

4. Each hidden neuron sums a weighted input signals and applying
   activation function. Then sends the result to all the neurons of the
   next layer.

5. Each output neuron sums weighted (the weights are bound to the neuron)
   incoming signals and calculates an output signal applying the activation
   function.

6. Each output neuron receives target value - the value of the output, which
   is correct for a given input signal, and calculates error, also
   calculates the amount by which to change the weight of relation. In addition
   it calculates the correction and sends offset to the neurons in the previous
   layer.

7. Each hidden neuron sums the incoming error from neurons in the subsequent
   layer and calculates the magnitude of the error. Magnitude of the error is
   the value obtained by multiplying the derivative of the activation function.
   Also calculates the amount by which to change the relation weight. In
   addition it calculates the offset correction.

8. Changing the weights.

   Each output neuron changes the weight of its relations with the displacement
   element and the hidden neurons. Each hidden neuron changes the weight of its
   relations with the displacement element and the output neurons.

9. Check the termination condition of the algorithm.

   The condition for the termination of the algorithm can be as achieving total
   square error at the output of the result of the network in advance a preset
   minimum during the learning process, as well as perform a number of
   iterations of the algorithm. The algorithm is based on a technique called
   gradient descent. Depending on the sign of the gradient of the function (in
   this case the value of the function - this is a mistake, and settings - is
   the weight of links in the network) gives the direction in which the values
   of increase (or decrease) the fastest.
