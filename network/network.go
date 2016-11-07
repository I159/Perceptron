package network

import (
	"errors"
	"github.com/I159/perceptron/neurons"
	"os"
)

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

func (network *Network) Perceive(file_path string) (error, *[][]byte) {
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
	/* TODO: prepare signal and pass to input neurons. */
}

func (network *Network) GetBackground(images []byte) byte {
	counter := make(map[byte]int)
	for _, i := range images {
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

/*func (network *Network) PrepareSignal() {
	TODO: get background and count difference between figure and background
}*/

/*func NewNetwork(image_x, image_y int) {
	TODO: Implement configurable network constructor.
}*/
