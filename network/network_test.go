package network

import (
	"math"
	"os"
	"testing"
)

const DIM_SIZE = 28
const TRAIN_SET_SIZE = 60000
const FILE_ADDRESS = "/home/i159/Downloads/train-images.idx3-ubyte"

var NETWORK = Network{}

func TestValidateFile(t *testing.T) {
	f, _ := os.Open(FILE_ADDRESS)
	file := ImagesFile{f}
	is_valid := file.IsValid()
	if is_valid == false {
		t.Fail()
	}
}

func TestDataLength(t *testing.T) {
	f, _ := os.Open(FILE_ADDRESS)
	file := ImagesFile{f}
	length := file.GetDataLength()
	if length != TRAIN_SET_SIZE {
		t.Fail()
	}
}

func TestImageSize(t *testing.T) {
	f, _ := os.Open(FILE_ADDRESS)
	file := ImagesFile{f}
	x, y := file.GetImageSize()

	if x != DIM_SIZE || y != DIM_SIZE {
		t.Errorf("X: %d. Y: %d.\n", x, y)
	}
}

func TestGetImage(t *testing.T) {
	f, _ := os.Open(FILE_ADDRESS)
	file := ImagesFile{f}
	image_length := uint32(math.Pow(DIM_SIZE, 2))

	image_bytes := file.GetImage(0, image_length)

	for _, i := range image_bytes {
		if i < 0 || i > 255 {
			t.Error(i)
		}
	}
}

func TestSplitImage(t *testing.T) {
	data_length := 60000
	_, images := SplitImages(FILE_ADDRESS)
	if len(*images) != data_length {
		t.Error(len(*images))
	}
}

func TestImageFileReadChunk(t *testing.T) {
	f, _ := os.Open(FILE_ADDRESS)
	file := ImagesFile{f}
	chunk := file.ReadChunk(0, 4)
	if len(chunk) != 4 {
		t.Fail()
	}
}

func TestGetBackground(t *testing.T) {
	_, images := SplitImages(FILE_ADDRESS)
	imgs := *images
	bg := GetBackground(imgs[0])
	counter := make(map[byte]int)
	for _, i := range imgs[0] {
		counter[i]++
		if counter[i] > counter[bg] {
			t.Fail()
		}
	}
}
