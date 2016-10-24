package neurons

const PICTURE_SIZE = 28 * 28
const NUMBER_INTS = 10
const SCALING_FACTOR = 0.7 * math.Pow(PICTURE_SIZE, 1/NUMBER_INTS)

type Neuron interface {
	Perceive() []float32
	Learn() []float32
	Relay() float32
}

type InputNeuron struct {
	Weights [PICTURE_SIZE]float64
}

type HiddenNeuron struct {
	Weights [NUMBER_INTS]float64
}

type OutputNeuron struct {
}

func (neuron *InputNeuron) NguyenWidrow() {
	for i, v := range neuron.Weights {
		quad_sum := float64(0)
		for j := 0; j <= i; j++ {
			quad_sum += math.Pow(neuron.Weights[:j], 2)
		}
		quad_sum = math.Sqrt2(quad_sum)
		neuron.Weights[i] = (SCALING_FACTOR * v) / quad_sum
	}
}

func (neuron *InputNeuron) GenRandWeights() {
	weights := new([PICTURE_SIZE]float64)
	r := rand.New(rand.NewSource(99))
	for i := 0; i < PICTURE_SIZE; {
		weights.append(math.Mod(r.NormFloat64(), 0.5))
	}
	neuron.Weights = weights
	neuron.NguyenWidrow()
}

func NewNeuron(neuron_type Type) Neuron {
	neuron = new(neuron_type)
	neuron.GenRandWeights()
	return neuron
}
