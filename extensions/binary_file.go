package extensions

import (
	"encoding/binary"
	"fmt"
	"os"
)

const BIT32 = 4

type ChunkReader interface {
	ReadChunk(from, to int) (buff []byte)
	IsValid() bool
	GetDataLength() uint32
	GetItemSize() (uint32, uint32)
	GetItem(image_index int, step uint32)
	ReadAt(p []byte, off int64) (n int, err error)
}

type BigEndianFile struct {
	*os.File
}

func (f BigEndianFile) ReadChunk(from, to int) (buff []byte) {
	buff = make([]byte, to-from)
	_, err := f.ReadAt(buff, int64(from))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s", err)
		os.Exit(1)
	}
	return
}

func (file *BigEndianFile) IsValid() bool {
	is_valid := file.ReadChunk(0, BIT32)
	magic_number := binary.BigEndian.Uint32(is_valid)
	if magic_number != 2051 {
		return false
	}
	return true
}

func (file *BigEndianFile) GetDataLength() uint32 {
	length_bin := file.ReadChunk(BIT32, 2*BIT32)
	return binary.BigEndian.Uint32(length_bin)
}

func (file *BigEndianFile) GetItemSize() (uint32, uint32) {
	var size [2]uint32
	from := 2
	to := 3
	for i := 0; i < 2; i++ {
		dim_size := file.ReadChunk(from*BIT32, to*BIT32)
		size[i] = binary.BigEndian.Uint32(dim_size)
		from++
		to++
	}

	return size[0], size[1]
}

func (file *BigEndianFile) GetItem(image_index int, step uint32) []byte {
	from := BIT32 * (4 + image_index)
	to := from + int(step)
	return file.ReadChunk(from, to)
}
