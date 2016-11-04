package network

import (
	"encoding/binary"
	"os"
)

const BIT32 = 4

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
