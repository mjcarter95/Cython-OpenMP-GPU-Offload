import numpy as np
import time
from gpuomp import scale_np


def timeit(fn, x, n_runs=10):
    fn(x)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn(x)
    t1 = time.perf_counter()

    return (t1 - t0) / n_runs


def cpu_scale(x):
    x *= 2.0


def main():
    N = int(2e9)
    n_runs = 5

    x_cpu = np.ones(N, dtype=np.float32)
    x_gpu = x_cpu.copy()

    t_cpu = timeit(cpu_scale, x_cpu, n_runs)
    t_gpu = timeit(scale_np, x_gpu, n_runs)

    assert np.allclose(x_cpu, x_gpu)

    print(f"N = {N:,}")
    print(f"CPU NumPy time: {t_cpu*1e3:.3f} ms")
    print(f"GPU OpenMP time: {t_gpu*1e3:.3f} ms")
    print(f"Speedup: {t_cpu / t_gpu:.2f}×")


if __name__ == "__main__":
    main()
