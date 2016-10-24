package perceptron

import (
	"math"
	"math/rand"
)

const PICTURE_SIZE = 28 * 28
const NUMBER_INTS = 10

type Neuron interface {
	GenRandWeights()
	NguyenWiderow()
}

type InputNeuron struct {
	Weights [PICTURE_SIZE]float64
}

type HiddenNeuron struct {
	Weights [NUMBER_INTS]float64
}

type OutputNeuron struct {
}

func (neuron *InputNeuron) NguyenWidrow() {
	scaling_factor := 0.7 * math.Pow(PICTURE_SIZE, 1/NUMBER_INTS)
	for i, v := range neuron.Weights {
		quad_sum := float64(0)
		for j := 0; j <= i; j++ {
			quad_sum += math.Pow(neuron.Weights[j], 2)
		}
		quad_sum = math.Sqrt(quad_sum)
		neuron.Weights[i] = (scaling_factor * v) / quad_sum
	}
}

func (neuron *InputNeuron) GenRandWeights() {
	weights := new([PICTURE_SIZE]float64)
	r := rand.New(rand.NewSource(99))
	for i := 0; i < PICTURE_SIZE; i++ {
		weights[i] = math.Mod(r.NormFloat64(), 0.5)
	}
	neuron.Weights = *weights
	neuron.NguyenWidrow()
}

func NewNeuron() InputNeuron {
	neuron := InputNeuron{}
	neuron.GenRandWeights()
	return neuron
}
