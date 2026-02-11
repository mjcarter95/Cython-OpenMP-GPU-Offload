from ._version import __version__

from .gpu import (
    heavy_seq_np,
    heavy_omp_cpu_np,
    heavy_gpu_begin_np,
    heavy_gpu_apply_np,
    heavy_gpu_end_np,
)
