import numpy as np
import faiss
import time
import sys
import logging
import csv
import sys

from word import Word, load_words, words2vecs
from report import printEntry

from recall import generateIndex, measureRecall

k = 10
batch_size = 64

vecs = words2vecs(load_words(sys.stdin))
out = sys.stdout
w = csv.writer(sys.stdout, delimiter='\t')

def testRecall(qname, M, nbits):
    d = vecs.shape[1]
    # Generate truthIndex
    truthIndex = generateIndex('L2', d, 0, 0, vecs)
    targetIndex = generateIndex(qname, d, M, nbits, vecs)

    # Measure the recall@k with original vectors
    truthCnt, hitCnt = measureRecall(truthIndex, targetIndex, k, vecs)

    w.writerow([qname, d, M, nbits, k, f"{hitCnt/truthCnt:.5f}"])
    out.flush()

if __name__ == '__main__':
    #   - Q-type:   quantizer name (PQ, RQ, LSQ)
    #   - d:        dimension number
    #   - M:        module number to split
    #   - nbits:    bits number to represent a module
    #   - k:        k-NN's top k value
    #   - recall0:  recall@{k} for training vectors
    w.writerow(['Q-type', 'd', 'M', 'nbits', 'k', 'recall0'])
    out.flush()

    M = 16
    nbits = 8
    testRecall('L2',  M, nbits)
    testRecall('PQ',  M, nbits)
    #testRecall('OPQ', M, nbits)
    testRecall('RQ',  M, nbits)
    testRecall('LSQ', M, nbits)
