import sys
import numpy as np
import faiss
import logging

from word import Word, load_words, words2index
from report import printEntry

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("kNN")
logger.setLevel(logging.INFO)

k = 10
M = 16
nbits = 8

# load words from STDIN
words = load_words(sys.stdin)
dim = len(words[0].vec)
logger.debug(f"loaded {len(words)} words with {len(words[0].vec)} dim vector")

# index vectors with L2 norm
index = words2index(words, lambda d: faiss.IndexPQ(d, M, nbits))
logger.info(f"indexed {index.ntotal} vectors with ID in {index.ntotal * index.index.sa_code_size()}")

for i in range(len(words)):
    v = words[i].vec
    vecs = np.array([v], dtype=np.float32)
    D, I = index.search(vecs, k+1)
    wD = [np.sqrt(d) for d in D[0]] # needed for only IndexFlagL2
    #wD = [-d for d in D[0]] # needed for only IndexFlagIP
    #wD = [d for d in D[0]]
    wI = [words[i].id for i in I[0]]
    printEntry(sys.stdout, wI[0], wI[1:], wD[1:])
