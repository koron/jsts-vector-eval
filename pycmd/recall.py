import numpy as np
import faiss
import time
import sys
import logging
import csv
import sys

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("RECALL")
logger.setLevel(logging.INFO)

k = 10
batch_size = 64

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

def generateVecs(d, num, norm):
    vecs = np.random.rand(num, d).astype('float32')
    if norm:
        vecs = faiss.NormalizationTransform(d).apply(vecs)
    return vecs

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

def testRecall(qname, d, M, nbits, norm, trainNum=10000, validNum=20000):
    trainVecs = generateVecs(d, trainNum, norm)
    validVecs = generateVecs(d, validNum, norm)

    # Generate truthIndex
    logger.debug("generate indexes")
    truthIndex = generateIndex('L2', d, 0, 0, trainVecs)
    targetIndex = generateIndex(qname, d, M, nbits, trainVecs)

    # Measure the recall@k with train vectors
    logger.debug(f"calculate recall@{k} on train vectors")
    t0, h0 = measureRecall(truthIndex, targetIndex, k, trainVecs)

    # Measure the recall@k with validation vectors
    logger.debug(f"calculate recall@{k} on validation vectors")
    t1, h1 = measureRecall(truthIndex, targetIndex, k, validVecs)
    w.writerow([qname, d, M, nbits, norm, len(trainVecs), len(validVecs), k, f"{h1/t1:.5f}", f"{h0/t0:.5f}"])
    out.flush()

def batchRecall(qname, norm):
    testRecall(qname,  8, 4, 8, norm)
    testRecall(qname, 16, 4, 8, norm)
    testRecall(qname, 32, 4, 8, norm)
    testRecall(qname, 64, 4, 8, norm)

#   - Q-type:   quantizer name (PQ, RQ, LSQ)
#   - d:        dimension number
#   - M:        module number to split
#   - nbits:    bits number to represent a module
#   - norm:     normalize vectors
#   - T-num:    training vector number
#   - V-num:    validation vector number
#   - k:        k-NN's k
#   - recall:   recall@{k} for validation vectors
#   - recall0:  recall@{k} for training vectors
w.writerow(['Q-type', 'd', 'M', 'nbits', 'norm', 'T-num', 'V-num', 'k', 'recall', 'recall0'])
out.flush()

def batchQuantizers(norm):
    #batchRecall('L2')
    batchRecall('PQ', norm)
    batchRecall('OPQ', norm)
    batchRecall('RQ', norm)
    batchRecall('LSQ', norm)

batchQuantizers(False)
batchQuantizers(True)
