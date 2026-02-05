cdef extern from "gpu_kernel.cpp":
    void scale_gpu(float* x, int N)

import numpy as np
cimport numpy as np

def scale_np(np.ndarray[np.float32_t, ndim=1] arr):
    scale_gpu(<float*>arr.data, arr.size)
