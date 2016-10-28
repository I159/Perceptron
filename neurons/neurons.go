package perceptron

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
)

const SCALING_BASE = 0.7
const WEIGHTS_LIM = 0.5

type Neuroner interface {
	GenRandWeights()
	NguyenWiderow()
}

type Neuron struct {
	Weights []float64
}

func (neuron *Neuron) NguyenWidrow(picture_size float64, items_num float64) {
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
	neuron.NguyenWidrow(picture_size, items_num)
}

type InputNeuron struct {
	*Neuron
	ImageVector []float64
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func GetDataLength(file *os.File) (int, uint32) {
	current_offset := 4
	bin_data_type := make([]byte, 4)
	offset, err := file.ReadAt(bin_data_type, int64(current_offset))
	check(err)
	return current_offset + offset, binary.BigEndian.Uint32(bin_data_type)
}

func (i *InputNeuron) Perceive(file_path string) {
	f, err := os.Open(file_path)
	check(err)
	_, data_length := GetDataLength(f)

	fmt.Printf("Data type: %d\n", data_length)
}

func NewNeuron(picture_size, items_num float64) *Neuron {
	neuron := new(Neuron)
	neuron.GenRandWeights(picture_size, items_num)
	return neuron
}

func NewInputNeuron(picture_size, items_num float64) *InputNeuron {
	neuron := InputNeuron{NewNeuron(picture_size, items_num), *new([]float64)}
	return &neuron
}
