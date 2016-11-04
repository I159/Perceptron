package tests

import (
	"github.com/I159/perceptron/neurons"
	"testing"
)

const BYTES_IN_32BIT = 4

var LIMIT = Limit{0.5, -0.5}
var NEURON = neurons.NewNeuron(DIM_SIZE*DIM_SIZE, 10)

type Limit struct {
	Upper float64
	Lower float64
}

type Offset struct {
	Offset int
}

func (o Offset) DoSteps(steps int) int {
	return o.Offset * steps
}

func TestPerceptronWeightsLen(t *testing.T) {
	if len(NEURON.Weights) > DIM_SIZE*DIM_SIZE {
		t.Fail()
	}
	if len(NEURON.Weights) < DIM_SIZE*DIM_SIZE {
		t.Fail()
	}
}

func TestPerceptronWeightsValues(t *testing.T) {
	for _, i := range NEURON.Weights {
		if i > LIMIT.Upper {
			t.Fail()
		}
		if i < LIMIT.Lower {
			t.Fail()
		}
	}
}
