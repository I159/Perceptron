package tests

import (
	"github.com/I159/perceptron/neurons"
	"testing"
)

var NEURON = perceptron.NewNeuron(28*28, 10)

func TestPerceptronWeightsLen(t *testing.T) {
	if len(NEURON.Weights) > 28*28 {
		t.Fail()
	}
	if len(NEURON.Weights) < 28*28 {
		t.Fail()
	}
}

func TestPerceptronWeightsValues(t *testing.T) {
	for _, i := range NEURON.Weights {
		if i > 0.5 {
			t.Fail()
		}
		if i < -0.5 {
			t.Fail()
		}
	}
}
