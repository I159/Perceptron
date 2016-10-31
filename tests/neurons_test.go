package tests

import (
	"github.com/I159/perceptron/neurons"
	"os"
	"testing"
)

var NEURON = perceptron.NewNeuron(28*28, 10)
var INPUT_NEURON = perceptron.NewInputNeuron(28*28, 10)

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

func TestPerceive(t *testing.T) {
	f, _ := os.Open("/home/i159/Downloads/train-images.idx3-ubyte")
	offset, length := perceptron.GetDataLength(4, f)
	if offset != 8 || length != 60000 {
		t.Fail()
	}
}

func TestImageSize(t *testing.T) {
	f, _ := os.Open("/home/i159/Downloads/train-images.idx3-ubyte")
	offset, x, y := perceptron.GetImageSize(8, f)
	if x != 28 || y != 28 || offset != 16 {
		t.Errorf("X: %d. Y: %d. Offset: %d\n", x, y, offset)
	}
}
