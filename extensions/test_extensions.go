package extensions

import (
	"math"
	"os"
	"testing"
)

const FILE_ADDRESS = "/home/i159/Downloads/train-images.idx3-ubyte"
const DIM_SIZE = 28
const TRAIN_SET_SIZE = 60000

func TestReadChnk(t *testing.T) {
	var res []byte
	f, _ := os.Open(FILE_ADDRESS)
	file := BigEndianFile{f}
	res = file.ReadChunk(0, 4)
	if len(res) != 4 {
		t.Fail()
	}
}

func TestValidateFile(t *testing.T) {
	f, _ := os.Open(FILE_ADDRESS)
	file := BigEndianFile{f}
	is_valid := file.IsValid()
	if is_valid == false {
		t.Error("File is not valid.")
	}
}

func TestDataLength(t *testing.T) {
	f, _ := os.Open(FILE_ADDRESS)
	file := BigEndianFile{f}
	length := file.GetDataLength()
	if length != TRAIN_SET_SIZE {
		t.Fail()
	}
}

func TestImageSize(t *testing.T) {
	f, _ := os.Open(FILE_ADDRESS)
	file := BigEndianFile{f}
	x, y := file.GetImageSize()

	if x != DIM_SIZE || y != DIM_SIZE {
		t.Errorf("X: %d. Y: %d.\n", x, y)
	}
}

func TestGetImage(t *testing.T) {
	f, _ := os.Open(FILE_ADDRESS)
	file := BigEndianFile{f}
	image_length := uint32(math.Pow(DIM_SIZE, 2))

	image_bytes := file.GetImage(0, image_length)

	for _, i := range image_bytes {
		if i < 0 || i > 255 {
			t.Error(i)
		}
	}
}
