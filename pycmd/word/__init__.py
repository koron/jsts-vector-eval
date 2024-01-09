import csv
import json
import numpy as np
import faiss

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

def words2vecs(words):
    return np.array([w.vec for w in words], dtype=np.float32)
