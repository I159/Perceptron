package tests

import (
	"github.com/I159/perceptron/neurons"
	"math"
	"os"
	"testing"
)

const BYTES_IN_32BIT = 4
const DIM_SIZE = 28
const TRAIN_SET_SIZE = 60000
const FILE_ADDRESS = "/home/i159/Downloads/train-images.idx3-ubyte"

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

var OFFSET = Offset{BYTES_IN_32BIT}
var LIMIT = Limit{0.5, -0.5}
var NEURON = perceptron.NewNeuron(DIM_SIZE*DIM_SIZE, 10)
var INPUT_NEURON = perceptron.NewInputNeuron(DIM_SIZE*DIM_SIZE, 10)

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

func TestValidateFile(t *testing.T) {
	f, _ := os.Open(FILE_ADDRESS)
	offset, is_valid := perceptron.IsValidBinFile(OFFSET.DoSteps(0), f)
	if offset != 4 || is_valid == false {
		t.Fail()
	}
}

func TestDataLength(t *testing.T) {
	f, _ := os.Open(FILE_ADDRESS)
	offset, length := perceptron.GetDataLength(OFFSET.DoSteps(1), f)
	if offset != OFFSET.DoSteps(2) || length != TRAIN_SET_SIZE {
		t.Fail()
	}
}

func TestImageSize(t *testing.T) {
	f, _ := os.Open(FILE_ADDRESS)
	offset, x, y := perceptron.GetImageSize(OFFSET.DoSteps(2), f)

	if x != DIM_SIZE || y != DIM_SIZE || offset != OFFSET.DoSteps(4) {
		t.Errorf("X: %d. Y: %d. Offset: %d. Dummy offset: %d\n", x, y, offset, OFFSET.DoSteps(8))
	}
}

func TestGetImage(t *testing.T) {
	f, _ := os.Open(FILE_ADDRESS)
	image_length := uint32(math.Pow(DIM_SIZE, 2))

	offset, image_bytes := perceptron.GetImage(OFFSET.DoSteps(4), image_length, f)

	if offset-OFFSET.DoSteps(4) != DIM_SIZE*DIM_SIZE {
		t.Error(offset)
	}

	for _, i := range image_bytes {
		if i < 0 || i > 255 {
			t.Error(i)
		}
	}
}

func TestPerceive(t *testing.T) {
	data_length := 60000
	_, images := INPUT_NEURON.Perceive(FILE_ADDRESS)
	if len(*images) != data_length {
		t.Error(len(*images))
	}
}
