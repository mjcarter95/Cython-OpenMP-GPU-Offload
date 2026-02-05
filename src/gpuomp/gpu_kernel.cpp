#include <omp.h>

extern "C" void scale_gpu(float* x, int N) {
  #pragma omp target teams distribute parallel for map(tofrom:x[0:N])
  for (int i = 0; i < N; ++i) {
    x[i] *= 2.0f;
  }
}
