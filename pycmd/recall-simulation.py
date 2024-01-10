import numpy as np
import faiss
import time
import sys
import logging
import csv
import sys

from recall import testRecall

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("RECALL")
logger.setLevel(logging.INFO)

k = 10
batch_size = 64

out = sys.stdout
w = csv.writer(sys.stdout, delimiter='\t')

def batchRecall(qname):
    testRecall(qname, 768, 16, 8, True, trainNum=2731, validNum=2731)

if __name__ == '__main__':
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

    batchRecall('L2')
    batchRecall('PQ')
    #batchRecall('OPQ')
    batchRecall('RQ')
    batchRecall('LSQ')
