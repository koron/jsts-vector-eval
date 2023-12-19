/*
Package embdjsts handles JSTS with embeddings.
*/
package embdjsts

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"io"
	"log"
	"slices"
	"strconv"

	"github.com/koron/jsts-vector-eval/internal/wordvec"
)

type Entry struct {
	ID          string
	Text1       string
	Text2       string
	Score       float64
	Embeddings1 []float64
	Embeddings2 []float64
}

func parseJSONFloatArray(s string) ([]float64, error) {
	var array []float64
	err := json.Unmarshal([]byte(s), &array)
	if err != nil {
		return nil, err
	}
	return array, nil
}

// ReadAll read all JSTS entries from r io.Reader.
func ReadAll(r io.Reader) ([]Entry, error) {
	r2 := csv.NewReader(r)
	r2.Comma = '\t'
	r2.FieldsPerRecord = 6
	var entries []Entry
	for {
		record, err := r2.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
		id, t1, t2 := record[0], record[1], record[2]
		rsc, re1, re2 := record[3], record[4], record[5]
		// parse score
		sc, err := strconv.ParseFloat(rsc, 64)
		if err != nil {
			log.Printf("WARN: failed to parse %q as score: %s", rsc, err)
		}
		e1, err := parseJSONFloatArray(re1)
		if err != nil {
			log.Printf("WARN: failed to parse embeddings#1: %s", err)
		}
		e2, err := parseJSONFloatArray(re2)
		if err != nil {
			log.Printf("WARN: failed to parse embeddings#2: %s", err)
		}
		entries = append(entries, Entry{
			ID:          id,
			Text1:       t1,
			Text2:       t2,
			Score:       sc,
			Embeddings1: e1,
			Embeddings2: e2,
		})
	}
	return entries, nil
}

// ToWords converts array of Entry to array of wordvec.Word
func ToWords(src []Entry) []wordvec.Word {
	dst := make([]wordvec.Word, 0, len(src)*2)
	seen := map[string]wordvec.Word{}
	for _, e := range src {
		id1 := e.ID + "_1"
		if w, ok := seen[e.Text1]; ok {
			if slices.Compare(w.Vec, e.Embeddings1) != 0 {
				log.Printf("WARN: conflicted vectors between %s and %s", w.ID, id1)
			}
			continue
		}
		w1 := wordvec.Word{
			ID:   id1,
			Text: e.Text1,
			Vec:  e.Embeddings1,
		}
		seen[e.Text1] = w1
		dst = append(dst, w1)

		id2 := e.ID + "_2"
		if w, ok := seen[e.Text2]; ok {
			if slices.Compare(w.Vec, e.Embeddings2) != 0 {
				log.Printf("WARN: conflicted vectors between %s and %s", w.ID, id2)
			}
			continue
		}
		w2 := wordvec.Word{
			ID:   id2,
			Text: e.Text2,
			Vec:  e.Embeddings2,
		}
		seen[e.Text2] = w2
		dst = append(dst, w2)
	}
	return dst
}
