package perceptron

import (
	"encoding/binary"
	"errors"
	//	"fmt"
	"math"
	"math/rand"
	"os"
)

const SCALING_BASE = 0.7
const WEIGHTS_LIM = 0.5
const BIT32 = 4

/*TODO: Include Perceive method to a common Neuroner interface. All the neurons
* should be able to perceive .*/
/* TODO: implement Perceive method for all the neuron types */
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

/* TODO: use separate file for binary operations with data. */
type ImagesFile struct {
	*os.File
}

func (f *ImagesFile) ReadChunk(from, to int) []byte {
	buff := make([]byte, to-from)
	_, err := f.ReadAt(buff, int64(from))
	check(err)
	return buff
}

func (file *ImagesFile) IsValid() bool {
	is_valid := file.ReadChunk(0, BIT32)
	magic_number := binary.BigEndian.Uint32(is_valid)
	if magic_number != 2051 {
		return false
	}
	return true
}

func (file *ImagesFile) GetDataLength() uint32 {
	length_bin := file.ReadChunk(BIT32, 2*BIT32)
	return binary.BigEndian.Uint32(length_bin)
}

func (file *ImagesFile) GetImageSize() (uint32, uint32) {
	x_dim := file.ReadChunk(2*BIT32, 3*BIT32)
	x_dim_size := binary.BigEndian.Uint32(x_dim)

	y_dim := file.ReadChunk(3*BIT32, 4*BIT32)
	y_dim_size := binary.BigEndian.Uint32(y_dim)

	return x_dim_size, y_dim_size
}

func (file *ImagesFile) GetImage(image_index int, step uint32) []byte {
	from := BIT32 * (4 + image_index)
	to := from + int(step)
	image := file.ReadChunk(from, to)
	return image
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
