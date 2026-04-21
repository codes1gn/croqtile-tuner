#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    // Dummy kernel placeholder
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaError_t err = cudaEventElapsedTime(&ms, start, stop);
    printf("cudaEventElapsedTime error: %s\n", cudaGetErrorString(err));
    printf("Elapsed: %f ms\n", ms);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
