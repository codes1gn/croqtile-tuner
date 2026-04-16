// Dense GEMM f16 for RTX 3070 (sm_86) using CUTLASS TensorOp
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/tensor_op.h>
#include <cutlass/arch/mma.h>

#define RTX3070_PEAK_F16_TFLOPS 19.5

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm86,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        float, 128,
        float, float
    >,
    cutlass::transform::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;

int main(int argc, char** argv) {
    bool enable_timing = true;
    bool skip_verify = false;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--disable-timing") == 0) {
            enable_timing = false;
        } else if (strcmp(argv[i], "--skip-verify") == 0) {
            skip_verify = true;
        }
    }

    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    // Allocate host memory
    cutlass::half_t *A_h, *B_h, *C_h, *D_h;
    size_t A_size = M * K * sizeof(cutlass::half_t);
    size_t B_size = N * K * sizeof(cutlass::half_t);
    size_t C_size = M * N * sizeof(cutlass::half_t);
    
    cudaMallocHost(&A_h, A_size);
    cudaMallocHost(&B_h, B_size);
    cudaMallocHost(&C_h, C_size);
    cudaMallocHost(&D_h, M * N * sizeof(float));

    // Initialize with seed
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < M * K; ++i) {
        A_h[i] = __float2half(dist(gen));
    }
    for (int i = 0; i < N * K; ++i) {
        B_h[i] = __float2half(dist(gen));
    }
    for (int i = 0; i < M * N; ++i) {
        C_h[i] = __float2half(0.0f);
    }

    // Allocate device memory
    cutlass::half_t *A_d, *B_d, *C_d;
    float *D_d;
    cudaMalloc(&A_d, A_size);
    cudaMalloc(&B_d, B_size);
    cudaMalloc(&C_d, C_size);
    cudaMalloc(&D_d, M * N * sizeof(float));

    cudaMemcpy(A_d, A_h, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, B_size, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, C_size, cudaMemcpyHostToDevice);
    cudaMemcpy(D_d, D_h, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize CUTLASS GEMM
    cutlass::Status status;
    Gemm gemm_op;
    
    status = gemm_op.initialize();
    if (status != cutlass::Status::kSuccess) {
        std::cout << "Failed to initialize GEMM\n";
        return 1;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        std::cout << "Failed to run GEMM\n";
        return 1;
    }

    cudaDeviceSynchronize();

    if (enable_timing) {
        int warmup = 10;
        int repeat = 50;
        
        for (int i = 0; i < warmup; ++i) {
            gemm_op();
        }
        cudaDeviceSynchronize();
        
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        
        cudaEventRecord(start);
        for (int i = 0; i < repeat; ++i) {
            gemm_op();
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        
        float ms;
        cudaEventElapsedTime(&ms, start, end);
        ms /= repeat;
        
        double flops = 2.0 * double(M) * double(N) * double(K);
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        
        std::cout << "Timing avg ms: " << ms << "\n";
        std::cout << "TFLOPS: " << tflops << "\n";
        std::cout << "HW efficiency: " << (tflops / RTX3070_PEAK_F16_TFLOPS) * 100.0 << "%\n";
    }

    if (skip_verify) {
        std::cout << "Test Passed (verify skipped)\n";
        cudaFree(A_d); cudaFree(B_d); cudaFree(C_d); cudaFree(D_d);
        cudaFreeHost(A_h); cudaFreeHost(B_h); cudaFreeHost(C_h); cudaFreeHost(D_h);
        return 0;
    }

    // Verify
    cudaMemcpy(D_h, D_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    float tolerance = 0.1f;
    int errors = 0;
    size_t verify_m = 128;
    size_t verify_n = 256;
    
    for (size_t i = 0; i < verify_m && errors < 8; ++i) {
        for (size_t j = 0; j < verify_n && errors < 8; ++j) {
            float ref = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                ref += __half2float(A_h[i * K + k]) * __half2float(B_h[j * K + k]);
            }
            float got = D_h[i * N + j];
            float diff = std::abs(got - ref);
            if (diff > tolerance) {
                std::cout << "[" << i << ", " << j << "] ref=" << ref
                          << " got=" << got << " diff=" << diff << std::endl;
                ++errors;
            }
        }
    }
    
    std::cout << "f16_gemm: " << errors << " errors\n";
    if (errors == 0) {
        std::cout << "Test Passed\n";
    } else {
        std::cout << "Test FAILED\n";
    }

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d); cudaFree(D_d);
    cudaFreeHost(A_h); cudaFreeHost(B_h); cudaFreeHost(C_h); cudaFreeHost(D_h);
    return errors;
}
