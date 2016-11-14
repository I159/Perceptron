package network

import (
	"errors"
	"github.com/I159/perceptron/neurons"
	"math"
	"os"
)

const EIGHT_BIT = 255

func check(e error) {
	if e != nil {
		panic(e)
	}
}

type Network struct {
	InputLayer   []neurons.Neuroner
	HiddenLayers [][]neurons.Neuroner
	OutputLayer  []neurons.Neuroner
}

/* TODO: implement dependency injection and test atomically */

func SplitImages(file_path string) (error, *[][]byte) {
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

func GetBackground(image []byte) byte {
	counter := make(map[byte]int)
	for _, i := range image {
		for k, _ := range counter {
			if k == i {
				counter[k]++
			} else {
				counter[i] = 1
			}
		}
	}
	var max = struct {
		Byte  byte
		Count int
	}{}
	for k, v := range counter {
		if v > max.Count {
			max.Byte = k
			max.Count = v
		}
	}
	return max.Byte
}

func PrepareSignal(image []byte) []float64 {
	background := GetBackground(image)
	signal := []float64{}
	for _, i := range image {
		difference := math.Abs(float64(background-i)) / EIGHT_BIT
		signal = append(signal, difference)
	}
	return signal
}

func (n *Network) Initialize(bin_file_addr string) {
	err, images := SplitImages(bin_file_addr)
	check(err)
	for _, img := range *images {
		for i, pixel := range PrepareSignal(img) {
			/* Transmit signal to input layer of neurons */
			n.InputLayer[i].Perceive(pixel)
		}
	}
}

func (n *Network) Classify(bin_file_addr string) rune {
	n.Initialize(bin_file_addr) /* Init downstream pipeline */
	for {
		select {
		case decision := <-neurons.Output:
			return decision
		}
	}
}

func NewNetwork(image_x, image_y int) {
	//input_layer := make([]neurons)
}
