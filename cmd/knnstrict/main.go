package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"slices"
	"sort"

	"github.com/koron/jsts-vector-eval/internal/wordvec"
)

type Cand struct {
	ID   string
	Dist float64
}

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

// InnerProduct is a distance function use negative inner product
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

func findKNN(words []wordvec.Word, target wordvec.Word, k int, dist func([]float64, []float64) float64) []Cand {
	rank := make([]Cand, 0, k+1)
	for _, w := range words {
		c := Cand{
			ID:   w.ID,
			Dist: dist(target.Vec, w.Vec),
		}
		x := sort.Search(len(rank), func(i int) bool {
			return c.Dist < rank[i].Dist
		})
		if x < k-1 {
			rank = slices.Insert(rank, x, c)
			if len(rank) > k {
				rank = rank[0:k]
			}
		}
	}
	return rank
}

var (
	distType string
	distFunc func([]float64, []float64) float64
	k        int
)

func writeAllKNN(w io.Writer, words []wordvec.Word, k int) error {
	w2 := csv.NewWriter(w)
	w2.Comma = '\t'
	record := make([]string, k+1)
	for _, w := range words {
		rank := findKNN(words, w, k+1, distFunc)
		rank = rank[1:]
		record[0] = w.ID
		for j, r := range rank {
			record[j+1] = fmt.Sprintf("%s:%f", r.ID, r.Dist)
		}
		if err := w2.Write(record); err != nil {
			return err
		}
	}
	w2.Flush()
	return w2.Error()
}

func main() {
	flag.StringVar(&distType, "d", "euclid", `disntace function: euclid, inner, cosine`)
	flag.IntVar(&k, "k", 10, `k value for top of k-NN`)
	flag.Parse()

	switch distType {
	case "euclid":
		distFunc = Euclid
	case "inner":
		distFunc = InnerProduct
	case "cosine":
		distFunc = Cosine
	default:
		log.Fatalf("unknown distance function: %s", distType)
	}

	words, err := wordvec.ReadAll(os.Stdin)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("loaded %d enties from input", len(words))

	err = writeAllKNN(os.Stdout, words, k)
	if err != nil {
		log.Fatal(err)
	}
}
