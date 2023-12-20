package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"slices"
	"sort"

	"github.com/koron/jsts-vector-eval/internal/distfunc"
	"github.com/koron/jsts-vector-eval/internal/wordvec"
)

type Cand struct {
	ID   string
	Dist float64
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

	distFunc = distfunc.ByName(distType)

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
