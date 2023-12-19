package main

import (
	"log"
	"os"

	"github.com/koron/jsts-vector-eval/internal/embdjsts"
	"github.com/koron/jsts-vector-eval/internal/wordvec"
)

// Convert JSTS+Embeddings file to wordvec file.

func main() {
	entries, err := embdjsts.ReadAll(os.Stdin)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("loaded %d enties from input", len(entries))
	words := embdjsts.ToWords(entries)
	log.Printf("extracted %d words", len(words))
	err = wordvec.WriteAll(os.Stdout, words)
	if err != nil {
		log.Fatal(err)
	}
}
