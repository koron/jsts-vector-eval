package distfunc

import (
	"log"
	"math"
)

// Euclid is a distance function uses euclid distance
func Euclid(v1, v2 []float64) float64 {
	var sum float64 = 0
	if len(v1) != len(v2) {
		panic("vector length unmatched")
	}
	for i := range v1 {
		sum += (v1[i] - v2[i]) * (v1[i] - v2[i])
	}
	return math.Sqrt(sum)
}

// InnerProduct is a distance function uses negative inner product
func InnerProduct(v1, v2 []float64) float64 {
	var sum float64 = 0
	if len(v1) != len(v2) {
		panic("vector length unmatched")
	}
	for i := range v1 {
		sum += v1[i] * v2[i]
	}
	return -sum
}

// Cosine is a distance function uses cosine distance
func Cosine(v1, v2 []float64) float64 {
	if len(v1) != len(v2) {
		panic("vector length unmatched")
	}
	var (
		sum float64
		l1  float64
		l2  float64
	)
	for i := range v1 {
		sum += v1[i] * v2[i]
		l1 += v1[i] * v1[i]
		l2 += v2[i] * v2[i]

	}
	return 1.0 - sum/(math.Sqrt(l1)*math.Sqrt(l2))
}

// ByName select a distnace function by name
func ByName(name string) func([]float64, []float64) float64 {
	switch name {
	case "euclid":
		return Euclid
	case "inner":
		return InnerProduct
	case "cosine":
		return Cosine
	default:
		log.Fatalf("unknown distance function: %s", name)
		return nil
	}
}
