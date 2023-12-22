import numpy as np
import faiss

d = 32
M = 4
nbits = 8

normalizer = faiss.NormalizationTransform(d)

nt = 10000
xt = np.random.rand(nt, d).astype('float32')
xt = normalizer.apply(xt)

n = 20000
x = np.random.rand(n, d).astype('float32')
x = normalizer.apply(x)

qz = faiss.ProductQuantizer(d, M, nbits)
#qz = faiss.ResidualQuantizer(d, M, nbits)
qz.train(xt)
codes = qz.compute_codes(x)
x2 = qz.decode(codes)

avg_relative_err = ((x - x2) ** 2).sum() / (x ** 2).sum()
print(f"PQ: d={d} M={M} nbits={nbits}")
#print(f"RQ: d={d} M={M} nbits={nbits} code_size={qz.code_size}")
print(f"avg_relative_err={avg_relative_err}")
