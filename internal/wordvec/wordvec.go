package wordvec

import (
	"encoding/csv"
	"encoding/json"
	"io"
)

type Word struct {
	ID   string
	Text string
	Vec  []float64
}

func WriteAll(w io.Writer, words []Word) error {
	w2 := csv.NewWriter(w)
	w2.Comma = '\t'
	for _, word := range words {
		b, err := json.Marshal(word.Vec)
		if err != nil {
			return err
		}
		err = w2.Write([]string{word.ID, word.Text, string(b)})
		if err != nil {
			return err
		}
	}
	return nil
}
