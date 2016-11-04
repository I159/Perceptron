package perceptron

import (
	"errors"
	//	"fmt"
	"math"
	"math/rand"
	"os"
)

const SCALING_BASE = 0.7
const WEIGHTS_LIM = 0.5

func check(e error) {
	if e != nil {
		panic(e)
	}
}

type Neuroner interface {
	GenRandWeights()
	NguyenWiderow()
}

type Neuron struct {
	Weights []float64
}

type InputNeuron struct {
	*Neuron
	ImageVector []float64
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

func (i *InputNeuron) Perceive(file_path string) (error, *[][]byte) {
	var invalid_file_error error
	images := new([][]byte)

	f, err := os.Open(file_path)
	check(err)
	file := ImagesFile{f}

	is_valid := file.IsValid()

	if is_valid == false {
		invalid_file_error = errors.New("Invalid mnist file.")
		return invalid_file_error, images
	}

	data_length := file.GetDataLength()
	x_size, y_size := file.GetImageSize()
	for i := 0; i < int(data_length); i++ {
		image := file.GetImage(i, x_size*y_size)
		*images = append(*images, image)
	}
	return invalid_file_error, images
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
