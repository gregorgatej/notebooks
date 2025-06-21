import cupy as cp
import numpy as np
from cupy.cuda import nvtx

doc_embeddings_gpu = cp.load("../doc_embeddings_dbpe.npy")
query_embedding_gpu = cp.load("../query_embedding_dbpe.npy")

def get_distance(item):
    return item[0]

def knn_search_verbose_cupy(X_data, query, k=3):
    nvtx.RangePush("knn_vectorized")
    diff = X_data - query
    squared = diff ** 2
    summed = cp.sum(squared, axis=1)
    dists = cp.sqrt(summed)
    nvtx.RangePop()

    nvtx.RangePush("knn_sort_and_topk")
    top_k_indices = cp.argsort(dists)[:k]
    nvtx.RangePop()

    nvtx.RangePush("knn_unpack_indices")
    indices = top_k_indices.tolist()
    nvtx.RangePop()

    return indices

cp.cuda.Device(0).synchronize()
nvtx.RangePush("knn_search_total")

top_matches = knn_search_verbose_cupy(doc_embeddings_gpu, query_embedding_gpu, k=3)

nvtx.RangePop()
cp.cuda.Device(0).synchronize()