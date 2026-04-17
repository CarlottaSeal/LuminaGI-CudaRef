// hello_cuda — smoke test for CUDA toolchain.
// Launches one block of 32 threads (one warp), each thread writes its index
// into a device buffer. Host prints the result.

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t _err = (call);                                       \
        if (_err != cudaSuccess) {                                       \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(_err));  \
            std::exit(1);                                                \
        }                                                                \
    } while (0)

__global__ void write_indices(int* out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) out[tid] = tid * tid;
}

int main()
{
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("GPU: %s (SM %d.%d, %d SMs, %zu MB)\n",
                prop.name, prop.major, prop.minor,
                prop.multiProcessorCount,
                prop.totalGlobalMem >> 20);

    constexpr int N = 32;
    int* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, N * sizeof(int)));

    write_indices<<<1, N>>>(d_buf, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_buf[N]{};
    CUDA_CHECK(cudaMemcpy(h_buf, d_buf, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_buf));

    std::printf("squares: ");
    for (int i = 0; i < N; ++i) std::printf("%d ", h_buf[i]);
    std::printf("\nOK\n");
    return 0;
}
