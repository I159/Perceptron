package neurons

import (
	//	"fmt"
	"math"
	"math/rand"
)

const SCALING_BASE = 0.7
const WEIGHTS_LIM = 0.5

/* TODO: channels for transmitting signals.
* Initial -> hidden channel: blocking unbuffered.
* Hidden -> hidden -> output channel: Ouroboros async channel to broadcast
signals.
* Output: channel to receive network decisions.
*/

var Output chan rune

type Neuroner interface {
	GenRandWeights(picture_size float64, items_num float64)
	NguyenWiderow(picture_size float64, items_num float64)
	Perceive(float64)
}

type Neuron struct {
	Weights []float64
}

func (neuron *Neuron) NguyenWiderow(picture_size float64, items_num float64) {
	scaling_factor := SCALING_BASE * math.Pow(picture_size, 1.0/items_num)
	quad_sum := float64(0)
	for _, j := range neuron.Weights {
		quad_sum += math.Pow(j, 2)
	}
	quad_sum = math.Sqrt(quad_sum)
	for i, v := range neuron.Weights {
		neuron.Weights[i] = (scaling_factor * v) / quad_sum
	}
}

func (neuron *Neuron) GenRandWeights(picture_size float64, items_num float64) {
	weights := []float64{}
	r := rand.New(rand.NewSource(99))
	for i := 0.0; i < picture_size; i++ {
		weights = append(weights, math.Mod(r.NormFloat64(), WEIGHTS_LIM))
	}
	neuron.Weights = weights
	neuron.NguyenWiderow(picture_size, items_num)
}

type InputNeuron struct {
	Neuron
}

func (neuron *InputNeuron) Perceive(float64) {
	/* TODO: Perceive method receives weighted or initial signals from a channel
	* of previous layer. */
}

func NewNeuron(picture_size, items_num float64) Neuroner {
	neuron := InputNeuron{Neuron{}}
	neuron.GenRandWeights(picture_size, items_num)
	return &neuron
}
