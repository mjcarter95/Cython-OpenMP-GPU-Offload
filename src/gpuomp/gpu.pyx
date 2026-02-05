# distutils: language = c++

cdef extern from "gpu_kernel.cpp":
    void heavy_seq(float* x, long long N, int iters)
    void heavy_omp_cpu(float* x, long long N, int iters)

    void heavy_gpu_begin(float* x, long long N)
    int  heavy_gpu_apply(float* x, long long N, int iters)
    void heavy_gpu_end(float* x, long long N)

import numpy as np
cimport numpy as np

def heavy_seq_np(np.ndarray[np.float32_t, ndim=1] arr, int iters):
    cdef long long N = arr.size
    heavy_seq(<float*>arr.data, N, iters)

def heavy_omp_cpu_np(np.ndarray[np.float32_t, ndim=1] arr, int iters):
    cdef long long N = arr.size
    heavy_omp_cpu(<float*>arr.data, N, iters)

def heavy_gpu_begin_np(np.ndarray[np.float32_t, ndim=1] arr):
    cdef long long N = arr.size
    heavy_gpu_begin(<float*>arr.data, N)

def heavy_gpu_apply_np(np.ndarray[np.float32_t, ndim=1] arr, int iters):
    cdef long long N = arr.size
    return heavy_gpu_apply(<float*>arr.data, N, iters)  # 0=GPU, 1=HOST

def heavy_gpu_end_np(np.ndarray[np.float32_t, ndim=1] arr):
    cdef long long N = arr.size
    heavy_gpu_end(<float*>arr.data, N)
