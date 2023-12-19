package wordvec

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"io"
	"log"
)

type Word struct {
	ID   string
	Text string
	Vec  []float64
}

// WriteAll writes all words []Word to w io.Writer
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
	w2.Flush()
	return w2.Error()
}

func parseJSONFloatArray(s string) ([]float64, error) {
	var array []float64
	err := json.Unmarshal([]byte(s), &array)
	if err != nil {
		return nil, err
	}
	return array, nil
}

// ReadAll read all Word from r io.Reader.
func ReadAll(r io.Reader) ([]Word, error) {
	r2 := csv.NewReader(r)
	r2.Comma = '\t'
	r2.FieldsPerRecord = 3
	var words []Word
	for {
		record, err := r2.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
		id, text, rvec := record[0], record[1], record[2]
		vec, err := parseJSONFloatArray(rvec)
		if err != nil {
			log.Printf("WARN: failed to parse vector: %s", err)
		}
		words = append(words, Word{
			ID:   id,
			Text: text,
			Vec:  vec,
		})
	}
	return words, nil
}
