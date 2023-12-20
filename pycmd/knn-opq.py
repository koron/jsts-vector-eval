import sys
import csv
import json
import numpy as np
import faiss
import logging

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("kNN")
logger.setLevel(logging.INFO)

class Word:
    def __init__(self, id, text, vec):
        self.id = id
        self.text = text
        self.vec = vec
    def __repr__(self):
        return f"Word(id={self.id}, text={self.text}, vec={self.vec[:5]}"

def load_words(r):
    words = []
    reader = csv.reader(r, delimiter='\t')
    for row in reader:
        id = row[0]
        text = row[1]
        data = json.loads(row[2])
        vec = np.array(data, dtype="float32")
        w = Word(id, text, vec)
        words.append(w)
    return words

def words2index(words, indexFactory):
    index = faiss.IndexIDMap(indexFactory(len(words[0].vec)))
    vecs = []
    ids = []
    for i in range(len(words)):
        vecs.append(words[i].vec)
        ids.append(i)
    vecs = np.array(vecs, dtype=np.float32)
    ids = np.array(ids, dtype=np.int64)
    if not index.is_trained:
        index.train(vecs)
    index.add_with_ids(vecs, ids)
    return index

def printEntry(w, qid, ids, dists):
    w.write(f"{qid}")
    for i in range(len(ids)):
        w.write("\t{}:{:f}".format(ids[i], dists[i]))
    w.write("\n")

k = 10
M = 16
nbits = 8

# load words from STDIN
words = load_words(sys.stdin)
dim = len(words[0].vec)
logger.info(f"loaded {len(words)} words with {len(words[0].vec)} dim vector")

# index vectors with L2 norm
#index = words2index(words, lambda d: faiss.IndexFlatIP(d))
index = words2index(words, lambda d: faiss.IndexPreTransform(faiss.OPQMatrix(d, M), faiss.IndexPQ(d, M, nbits)))
logger.info(f"indexed {index.ntotal} vectors with ID in {index.ntotal * index.index.sa_code_size()}")

for i in range(len(words)):
    v = words[i].vec
    vecs = np.array([v], dtype=np.float32)
    D, I = index.search(vecs, k+1)
    #wD = [np.sqrt(d) for d in D[0]] # needed for only IndexFlagL2
    #wD = [-d for d in D[0]] # needed for only IndexFlagIP
    wD = [d for d in D[0]]
    wI = [words[i].id for i in I[0]]
    printEntry(sys.stdout, wI[0], wI[1:], wD[1:])
