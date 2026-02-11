import os
import time
import numpy as np

from gpuomp import (
    heavy_seq_np,
    heavy_omp_cpu_np,
    heavy_gpu_begin_np,
    heavy_gpu_apply_np,
    heavy_gpu_end_np,
)

def timeit(fn, *args, n_runs=3):
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / n_runs

def reset(x):
    x.fill(0.9)

def main():
    N = 500_000
    iters = 200
    cpu_runs = 3
    gpu_kernel_repeats = 10

    print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS", "<unset>"))
    print("N =", f"{N:,}", "float32 (~%.2f GB)" % (N * 4 / 1e9))
    print("iters =", iters)
    print("gpu_kernel_repeats =", gpu_kernel_repeats)

    x = np.empty(N, dtype=np.float32)

    # (a) sequential CPU
    reset(x)
    t_seq = timeit(heavy_seq_np, x, iters, n_runs=cpu_runs)

    # (b) OpenMP shared-memory CPU
    reset(x)
    t_omp = timeit(heavy_omp_cpu_np, x, iters, n_runs=cpu_runs)

    # (c) GPU offload persistent data region
    reset(x)
    t_gpu_total = None
    gpu_loc = None

    try:
        t0 = time.perf_counter()
        heavy_gpu_begin_np(x)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        loc = heavy_gpu_apply_np(x, iters)
        for _ in range(gpu_kernel_repeats - 1):
            heavy_gpu_apply_np(x, iters)
        t3 = time.perf_counter()

        heavy_gpu_end_np(x)
        t4 = time.perf_counter()

        gpu_loc = "GPU" if loc == 0 else "HOST"
        t_gpu_total = t4 - t0
        t_gpu_kernel = t3 - t2
        t_gpu_copy = (t1 - t0) + (t4 - t3)

        print("\nGPU probe executed on:", gpu_loc)
        print("GPU copy time (enter+exit): %.3f ms" % (t_gpu_copy * 1e3))
        print("GPU kernel time (repeats):  %.3f ms" % (t_gpu_kernel * 1e3))
        print("GPU total time:             %.3f ms" % (t_gpu_total * 1e3))

    except Exception as e:
        print("\nGPU path raised an exception:", repr(e))

    print("\n=== Results ===")
    print(f"Sequential CPU:     {t_seq*1e3:.3f} ms   (baseline)")
    print(f"OpenMP CPU:         {t_omp*1e3:.3f} ms   (speedup {t_seq/t_omp:.2f}×)")

    if t_gpu_total is not None:
        print(f"OpenMP target ({gpu_loc}): {t_gpu_total*1e3:.3f} ms   (speedup {t_seq/t_gpu_total:.2f}×)")
        print(f"OpenMP target kernel-only: {t_gpu_kernel*1e3:.3f} ms (repeats={gpu_kernel_repeats})")
    else:
        print("OpenMP target (GPU): SKIPPED/FAILED")

if __name__ == "__main__":
    main()
