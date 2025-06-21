import cupy as cp
import numpy as np
from cupy.cuda import nvtx

doc_embeddings_gpu = cp.load("../doc_embeddings_dbpe.npy")
query_embedding_gpu = cp.load("../query_embedding_dbpe.npy")

l2_norm_kernel = cp.ElementwiseKernel(
    in_params='raw float32 x, raw float32 y, int32 dim',
    out_params='float32 result',
    operation='''
    float sum = 0.0;
    for (int j = 0; j < dim; ++j) {
        float diff = x[i * dim + j] - y[j];
        sum += diff * diff;
    }
    result = sqrtf(sum);  // fused sqrt here
    ''',
    name='fused_l2_norm'
)

def knn_search_fused_cupy(X_data, query, k=3):
    nvtx.RangePush("knn_fused_l2")
    n_rows, n_dims = X_data.shape
    dists = cp.empty(n_rows, dtype=cp.float32)
    l2_norm_kernel(X_data.ravel(), query, n_dims, dists)
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

top_matches = knn_search_fused_cupy(doc_embeddings_gpu, query_embedding_gpu, k=3)

nvtx.RangePop()
cp.cuda.Device(0).synchronize()

print("Top matches:", top_matches)