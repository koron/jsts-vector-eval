import numpy as np
import faiss
import time
import sys
import logging
import csv
import sys

from word import Word, load_words, words2vecs
from report import printEntry

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("RECALLWORDS")
logger.setLevel(logging.INFO)

k = 10
batch_size = 64

vecs = words2vecs(load_words(sys.stdin))
out = sys.stdout
w = csv.writer(sys.stdout, delimiter='\t')

def measureRecall(truthIndex, targetIndex, k, vecs):
    truthCount = 0 
    hitCount = 0
    for i in range(0, len(vecs), batch_size):
        qvecs = vecs[i : min(i+batch_size, len(vecs))]
        d0, truthIds = truthIndex.search(qvecs, k)
        d1, targetIds = targetIndex.search(qvecs, k)
        for j in range(0, len(truthIds)):
            truthCount += len(truthIds[j])
            hitCount += len(np.intersect1d(truthIds[j], targetIds[j]))
    return truthCount, hitCount

def generateIndex(qname, d, M, nbits, vecs):
    match qname:
        case 'L2':
            coreIndex = faiss.IndexFlatL2(d)
        case 'PQ':
            coreIndex = faiss.IndexPQ(d, M, nbits)
        case 'OPQ':
            coreIndex = faiss.IndexPreTransform(faiss.OPQMatrix(d, M), faiss.IndexPQ(d, M, nbits))
        case 'RQ':
            coreIndex = faiss.IndexResidualQuantizer(d, M, nbits)
        case 'LSQ':
            coreIndex = faiss.IndexLocalSearchQuantizer(d, M, nbits)
        case _:
            raise Exception(f"unknown quantizer: {qname}")
    index = faiss.IndexIDMap(coreIndex)
    if not index.is_trained:
        index.train(vecs)
    index.add_with_ids(vecs, list(range(0, len(vecs))))
    return index

def testRecall(qname, M, nbits):
    d = vecs.shape[1]
    # Generate truthIndex
    logger.debug("generate indexes")
    truthIndex = generateIndex('L2', d, 0, 0, vecs)
    targetIndex = generateIndex(qname, d, M, nbits, vecs)

    # Measure the recall@k with original vectors
    truthCnt, hitCnt = measureRecall(truthIndex, targetIndex, k, vecs)

    w.writerow([qname, d, M, nbits, k, f"{hitCnt/truthCnt:.5f}"])
    out.flush()

w.writerow(['Q-type', 'd', 'M', 'nbits', 'k', 'recall0'])
out.flush()

testRecall('L2',  16, 8)
testRecall('PQ',  16, 8)
testRecall('OPQ', 16, 8)
testRecall('RQ',  16, 8)
testRecall('LSQ', 16, 8)
