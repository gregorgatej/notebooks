import cupy as cp
import numpy as np
from cupy.cuda import nvtx

doc_embeddings_gpu = cp.load("../doc_embeddings_dbpe.npy")
query_embedding_gpu = cp.load("../query_embedding_dbpe.npy")

def l2_distance_verbose_cupy(data, query):
    nvtx.RangePush("l2_distance_verbose_cupy")
    diff = data - query
    squared = diff ** 2
    summed = cp.sum(squared)
    dist = cp.sqrt(summed)
    nvtx.RangePop()
    return dist

def get_distance(item):
    return item[0]

def knn_search_verbose_cupy(X_data, query, k=3):
    nvtx.RangePush("knn_search_loop")
    distances = []
    for i in range(len(X_data)):
        dist = l2_distance_verbose_cupy(X_data[i], query)
        distances.append((dist.item(), i))
    nvtx.RangePop()

    nvtx.RangePush("knn_sort_and_topk")
    distances.sort(key=get_distance)
    top_k = distances[:k]
    nvtx.RangePop()

    nvtx.RangePush("knn_unpack_indices")
    indices = []
    for item in top_k:
        _, idx = item
        indices.append(idx)
    nvtx.RangePop()

    return indices

cp.cuda.Device(0).synchronize()
nvtx.RangePush("knn_search_total")

top_matches = knn_search_verbose_cupy(doc_embeddings_gpu, query_embedding_gpu, k=3)

nvtx.RangePop()
cp.cuda.Device(0).synchronize()