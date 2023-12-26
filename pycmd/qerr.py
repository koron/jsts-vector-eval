import numpy as np
import faiss
import time
import sys
import logging

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("QERR")
logger.setLevel(logging.INFO)

def encodeAndDecode(qz, vecs):
    t0 = time.time()
    codes = qz.compute_codes(vecs)
    t1 = time.time()
    decodedVecs = qz.decode(codes)
    t2 = time.time()
    encodeMsec = int((t1 - t0) * 1000)
    decodeMsec = int((t2 - t1) * 1000)
    avgRelativeErr = ((vecs - decodedVecs) ** 2).sum() / (vecs ** 2).sum()
    return avgRelativeErr, encodeMsec, decodeMsec

def quantizerAvgRelError(qname, d, M, nbits, norm, trainNum=10000, validNum=20000):
    normalizer = faiss.NormalizationTransform(d)

    # Generate training vectors
    trainVecs = np.random.rand(trainNum, d).astype('float32')
    if norm:
        trainVecs = normalizer.apply(trainVecs)

    # Generate validation vectors
    validVecs = np.random.rand(validNum, d).astype('float32')
    if norm:
        validVecs = normalizer.apply(validVecs)

    # Create a quantizer
    match qname:
        case 'PQ':
            qz = faiss.ProductQuantizer(d, M, nbits)
        case 'RQ':
            qz = faiss.ResidualQuantizer(d, M, nbits)
        case 'LSQ':
            qz = faiss.LocalSearchQuantizer(d, M, nbits)
        case _:
            raise Exception(f"unknown quantizer: {qname}")
    logger.debug(f"{qname}: d={d} M={M} nbits={nbits} code_size={qz.code_size}")

    # Train a quantizer
    start = time.time()
    qz.train(trainVecs)
    trainMsec = int((time.time() - start) * 1000)

    # Calculate (average ralative) errors for validation vectors
    validErr, validEncMsec, validDecMsec = encodeAndDecode(qz, validVecs)
    logger.debug(f"validErr={validErr:f}")

    # Calculate errors for training vectors
    trainErr, trainEncMsec, trainDecMsec = encodeAndDecode(qz, trainVecs)
    logger.debug(f"trainErr={trainErr:f}")
    print(f"{qname}\t{d}\t{M}\t{nbits}\t{norm}\t{trainNum}\t{validNum}\t{qz.code_size}\t{trainMsec}\t{validEncMsec}\t{validDecMsec}\t{trainEncMsec}\t{trainDecMsec}\t{validErr:f}\t{trainErr:f}", flush=True)
    return (validErr, trainErr)

def testQNorm(q, norm):
    # The control
    quantizerAvgRelError(q, 32, 4, 8, norm)

    # M variation
    quantizerAvgRelError(q, 32,  2, 8, norm)
    quantizerAvgRelError(q, 32,  8, 8, norm)
    quantizerAvgRelError(q, 32, 16, 8, norm)

    # nbits variation
    quantizerAvgRelError(q, 32, 4, 7, norm)
    quantizerAvgRelError(q, 32, 4, 6, norm)
    quantizerAvgRelError(q, 32, 4, 5, norm)

    # d variations
    quantizerAvgRelError(q,  8, 4, 8, norm)
    quantizerAvgRelError(q, 16, 4, 8, norm)
    quantizerAvgRelError(q, 64, 4, 8, norm)

    # valid vector number variation
    quantizerAvgRelError(q, 32,  4, 8, norm, validNum=40000)
    quantizerAvgRelError(q, 32,  4, 8, norm, validNum=80000)
    quantizerAvgRelError(q, 32,  4, 8, norm, validNum=160000)

def testQ(q):
    testQNorm(q, False)
    testQNorm(q, True)

# Print the header
#   - Q-type:   quantizer name (PQ, RQ, LSQ)
#   - d:        dimension number
#   - M:        module number to split
#   - nbits:    bits number to represent a module
#   - T-num:    training vector number
#   - V-num:    validation vector number
#   - C-size:   code size to represent a vector
#   - T-ms:     elapsed millisecond for training
#   - VE-ms:    elapsed millisecond to encode validation vectors
#   - VD-ms:    elapsed millisecond to decode validation vectors
#   - TE-ms:    elapsed millisecond to encode training vectors
#   - TD-ms:    elapsed millisecond to decode training vectors
#   - V-err:    average relative error of decode and encode validation vectors
#   - T-err:    average relative error of decode and encode training vectors
print("Q-type\td\tM\tnbits\tnorm\tT-num\tV-num\tC-size\tT-ms\tVE-ms\tVD-ms\tTE-ms\tTD-ms\tV-err\tT-err", flush=True)

testQ('PQ')
testQ('RQ')
testQ('LSQ')
