import cupy as cp
import numpy as np

# Load data onto GPU
doc_embeddings_gpu   = cp.load("../doc_embeddings_dbpe.npy")
query_embedding_gpu  = cp.load("../query_embedding_dbpe.npy")
cp.cuda.Device(0).synchronize()

# Allocate pageable NumPy arrays on host
doc_embeddings_host  = np.empty(doc_embeddings_gpu.shape, dtype=doc_embeddings_gpu.dtype)
query_embedding_host = np.empty(query_embedding_gpu.shape, dtype=query_embedding_gpu.dtype)

# Perform synchronous cudaMemcpy (Device→Host)
doc_ptr, query_ptr = doc_embeddings_gpu.data.ptr, query_embedding_gpu.data.ptr
doc_nbytes, query_nbytes = doc_embeddings_gpu.nbytes, query_embedding_gpu.nbytes

kind = cp.cuda.runtime.memcpyDeviceToHost

cp.cuda.runtime.memcpy(
    doc_embeddings_host.ctypes.data,
    doc_ptr,
    doc_nbytes,
    kind
)
cp.cuda.runtime.memcpy(
    query_embedding_host.ctypes.data,
    query_ptr,
    query_nbytes,
    kind
)
