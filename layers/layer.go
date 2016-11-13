package layers

import "github.com/I159/perceptron/neurons"

type Layer struct {
	Transmitter chan float64
	Neurons     []neurons.Neuroner
	Signal      []float64
}

func (l *Layer) Get() {
}
