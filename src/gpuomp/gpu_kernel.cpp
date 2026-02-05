#include <omp.h>
#include <cstdint>

static inline float heavy_op(float v, int iters) {
  for (int k = 0; k < iters; ++k) {
    v = v * 1.000001f + 0.000001f;
    v = v * v + 0.1f;
    v = 0.5f * v + 0.25f;
  }
  return v;
}

// (a) Sequential CPU
extern "C" void heavy_seq(float* x, std::int64_t N, int iters) {
  for (std::int64_t i = 0; i < N; ++i) {
    x[i] = heavy_op(x[i], iters);
  }
}

// (b) Shared-memory CPU OpenMP
extern "C" void heavy_omp_cpu(float* x, std::int64_t N, int iters) {
  #pragma omp parallel for
  for (std::int64_t i = 0; i < N; ++i) {
    x[i] = heavy_op(x[i], iters);
  }
}

// (c) GPU Offload: persistent device data
extern "C" void heavy_gpu_begin(float* x, std::int64_t N) {
  #pragma omp target enter data map(to: x[0:N])
}

extern "C" int heavy_gpu_apply(float* x, std::int64_t N, int iters) {
  int on_host = 1;

  #pragma omp target data map(tofrom: x[0:N])
  {
    #pragma omp target map(tofrom: on_host)
    {
      on_host = omp_is_initial_device();
    }

    #pragma omp target teams distribute parallel for
    for (std::int64_t i = 0; i < N; ++i) {
      x[i] = heavy_op(x[i], iters);
    }
  }

  return on_host;
}

extern "C" void heavy_gpu_end(float* x, std::int64_t N) {
  #pragma omp target exit data map(from: x[0:N])
}
