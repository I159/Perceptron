package network

import (
	"github.com/I159/perceptron/extensions"
	"os"
	"testing"
)

const FILE_ADDRESS = "/home/i159/Downloads/train-images.idx3-ubyte"

var NETWORK = Network{}

func TestSplitImage(t *testing.T) {
	data_length := 60000
	_, images := SplitImages(FILE_ADDRESS)
	if len(*images) != data_length {
		t.Error(len(*images))
	}
}

func TestImageFileReadChunk(t *testing.T) {
	f, _ := os.Open(FILE_ADDRESS)
	file := extensions.BigEndianFile{f}
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

func TestPrepareSignal(t *testing.T) {
	_, images := SplitImages(FILE_ADDRESS)
	imgs := *images
	signal := PrepareSignal(imgs[0])
	for _, i := range signal {
		if i > 1 || i < 0 {
			t.Fail()
		}
	}
}

func TestInit(t *testing.T) {
}
