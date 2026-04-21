// iter027_sparse_unit_test.cu
// Unit test for sparse mma.sp m16n8k16 to understand exact fragment layout
// 
// Test approach:
// 1. Create a simple A matrix with known sparse pattern
// 2. Create a simple B matrix (all 1s or identity-like)
// 3. Run both scalar and mma.sp versions
// 4. Compare output element by element

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

// Single tile: m16 x n8 x k16 (logical K, physical sparse K = 8)

// Scalar reference on CPU
void cpu_sparse_matmul_tile(
    const half* A_packed,  // [16, 8] (compressed)
    const uint32_t* A_meta,  // [16] metadata
    const half* B,           // [8, 16] row-major (n=8, k=16)
    float* C,                // [16, 8] output
    int M, int N, int K      // 16, 8, 16
) {
    // For each output element
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            
            // Iterate over K dimension using metadata
            uint32_t meta = A_meta[m];
            for (int cg = 0; cg < K / 4; ++cg) {  // 4 groups of 4
                uint32_t nibble = (meta >> (cg * 4)) & 0xF;
                int idx0 = nibble & 0x3;
                int idx1 = (nibble >> 2) & 0x3;
                
                // Physical k positions in packed A
                int k_phys0 = cg * 2;
                int k_phys1 = cg * 2 + 1;
                
                // Logical k positions for B
                int k_log0 = cg * 4 + idx0;
                int k_log1 = cg * 4 + idx1;
                
                float a0 = __half2float(A_packed[m * (K/2) + k_phys0]);
                float a1 = __half2float(A_packed[m * (K/2) + k_phys1]);
                float b0 = __half2float(B[n * K + k_log0]);
                float b1 = __half2float(B[n * K + k_log1]);
                
                sum += a0 * b0 + a1 * b1;
            }
            C[m * N + n] = sum;
        }
    }
}

__device__ __forceinline__ void mma_sp_m16n8k16_f16_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    uint32_t meta
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta)
    );
#endif
}

// GPU kernel: single warp computes m16n8 tile
__global__ void sparse_mma_tile_kernel(
    const half* A_packed,   // [16, 8]
    const uint32_t* A_meta, // [16]
    const half* B,          // [8, 16]
    half* C                 // [16, 8]
) {
    const int laneId = threadIdx.x;
    const int groupID = laneId / 4;       // 0-7
    const int groupLaneID = laneId % 4;   // 0-3
    
    // Fragment layout for mma.sp m16n8k16:
    // A (sparse, m16 x k8 physical):
    //   - a[0,1]: row=groupID,   k=groupLaneID*2, groupLaneID*2+1
    //   - a[2,3]: row=groupID+8, k=groupLaneID*2, groupLaneID*2+1
    //
    // B (dense, n8 x k16):
    //   - b[0,1]: n=groupID,   k=groupLaneID*2, groupLaneID*2+1
    //   - b[2,3]: n=groupID,   k=groupLaneID*2+8, groupLaneID*2+9
    //   (Note: groupID maps to N, not separate halves)
    
    // Load A fragment
    int aRow0 = groupID;
    int aRow1 = groupID + 8;
    int aK = groupLaneID * 2;
    
    half aRegs[4];
    aRegs[0] = A_packed[aRow0 * 8 + aK];      // row groupID, k [0,2,4,6]
    aRegs[1] = A_packed[aRow0 * 8 + aK + 1];  // row groupID, k [1,3,5,7]
    aRegs[2] = A_packed[aRow1 * 8 + aK];      // row groupID+8, k [0,2,4,6]
    aRegs[3] = A_packed[aRow1 * 8 + aK + 1];  // row groupID+8, k [1,3,5,7]
    
    uint32_t a_u32[2];
    a_u32[0] = *reinterpret_cast<uint32_t*>(&aRegs[0]);
    a_u32[1] = *reinterpret_cast<uint32_t*>(&aRegs[2]);
    
    // Load B fragment
    // For m16n8k16 dense (non-sparse B):
    // groupID (0-7) maps to which of the 8 N columns
    int bN = groupID;  // Only 8 columns in N
    int bK0 = groupLaneID * 2;
    int bK1 = groupLaneID * 2 + 8;
    
    half bRegs[4];
    bRegs[0] = B[bN * 16 + bK0];      // n=groupID, k=[0,2,4,6]
    bRegs[1] = B[bN * 16 + bK0 + 1];  // n=groupID, k=[1,3,5,7]
    bRegs[2] = B[bN * 16 + bK1];      // n=groupID, k=[8,10,12,14]
    bRegs[3] = B[bN * 16 + bK1 + 1];  // n=groupID, k=[9,11,13,15]
    
    uint32_t b_u32[2];
    b_u32[0] = *reinterpret_cast<uint32_t*>(&bRegs[0]);
    b_u32[1] = *reinterpret_cast<uint32_t*>(&bRegs[2]);
    
    // Load metadata
    // For sparse MMA, metadata is per-row
    // But for the instruction, we need the metadata for the rows this thread contributes to
    // Since all threads in a group share the same row info...
    // Actually, the metadata is passed uniformly - all threads use the same meta for the warp
    
    // Wait - that can't be right. Let me check the PTX documentation again.
    // The metadata E is "32-bit sparse metadata for multiplicand A"
    // This suggests one u32 covers the entire sparse pattern for the tile being computed
    
    // For m16n8k16 sparse, the logical K=16 with 4 groups of 4 = 16 bits of metadata
    // But rows 0-15 each have their own sparsity pattern...
    
    // Reading the PTX doc more carefully:
    // "The sparsity pattern of A is encoded in 32-bit operand E. Each thread provides
    //  the 4-bit encoding of the sparsity pattern for the elements of A used in the 
    //  thread's calculation."
    
    // So each thread provides its own metadata! The 4-bit encoding is:
    // bits [3:2] = idx1, bits [1:0] = idx0
    // This covers one 4-element group.
    
    // But we have 4 groups (k16 / 4 = 4). So each thread provides 4 * 4 = 16 bits?
    // No wait, looking at the struct, it's just one u32 per MMA call.
    
    // Let me look at the CUTLASS code more carefully...
    // In SM80_SPARSE_16x8x16_F32F16F16F32_TN::fma, there's spsel=0x0 at the end
    // and 'e' is a single u32.
    
    // I think the metadata format is that threads 0-15 each provide metadata
    // for their respective M row. Let me use the row-based metadata:
    
    uint32_t meta = A_meta[aRow0];  // Row groupID's metadata
    
    // Actually wait, threads 16-31 would need different metadata
    // Let me try: the metadata might be broadcasted, or it might need
    // to be the correct one for each thread's row
    
    // Let me try a different approach: what if the spsel (selector) determines
    // which row's metadata to use?
    
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    mma_sp_m16n8k16_f16_f32(
        acc[0], acc[1], acc[2], acc[3],
        a_u32[0], a_u32[1],
        b_u32[0], b_u32[1],
        acc[0], acc[1], acc[2], acc[3],
        meta
    );
    
    // Store output
    // Output layout: each thread writes to (groupID, col) and (groupID+8, col)
    // col is determined by groupLaneID
    // d[0,1] -> row=groupID,   col=(groupLaneID%2)*2 + groupLaneID/2 * 4
    // d[2,3] -> row=groupID+8, same col
    
    int outCol = (groupLaneID % 2) * 2 + (groupLaneID / 2) * 4;
    // groupLaneID=0: col=0, groupLaneID=1: col=2, groupLaneID=2: col=4, groupLaneID=3: col=6
    
    C[groupID * 8 + outCol] = __float2half(acc[0]);
    C[groupID * 8 + outCol + 1] = __float2half(acc[1]);
    C[(groupID + 8) * 8 + outCol] = __float2half(acc[2]);
    C[(groupID + 8) * 8 + outCol + 1] = __float2half(acc[3]);
}

int main() {
    printf("Sparse MMA m16n8k16 Unit Test\n\n");
    
    // Create test data
    // A: 16x16 sparse (stored as 16x8 packed + 16 u32 metadata)
    // B: 8x16 dense
    
    std::vector<float> A_dense(16 * 16, 0.0f);
    std::vector<half> A_packed(16 * 8);
    std::vector<uint32_t> A_meta(16);
    std::vector<half> B(8 * 16);
    
    // Initialize with simple pattern: 2:4 sparsity, positions 0 and 1 always
    // Values: row index as value
    for (int m = 0; m < 16; ++m) {
        uint32_t meta = 0;
        for (int cg = 0; cg < 4; ++cg) {  // 4 groups
            int k_base = cg * 4;
            // Always use positions 0 and 1 in each group
            A_dense[m * 16 + k_base + 0] = float(m + 1);  // Position 0
            A_dense[m * 16 + k_base + 1] = float(m + 1);  // Position 1
            
            // Packed: store the two non-zero values
            A_packed[m * 8 + cg * 2 + 0] = __float2half(float(m + 1));
            A_packed[m * 8 + cg * 2 + 1] = __float2half(float(m + 1));
            
            // Metadata: idx0=0, idx1=1 -> nibble = 0 | (1 << 2) = 4
            meta |= (4 << (cg * 4));
        }
        A_meta[m] = meta;
    }
    
    // Initialize B: all ones for simplicity
    for (int i = 0; i < 8 * 16; ++i) {
        B[i] = __float2half(1.0f);
    }
    
    // CPU reference
    std::vector<float> C_ref(16 * 8, 0.0f);
    cpu_sparse_matmul_tile(A_packed.data(), A_meta.data(), B.data(), C_ref.data(), 16, 8, 16);
    
    printf("CPU Reference output (first few rows):\n");
    for (int m = 0; m < 4; ++m) {
        printf("Row %d: ", m);
        for (int n = 0; n < 8; ++n) {
            printf("%.1f ", C_ref[m * 8 + n]);
        }
        printf("\n");
    }
    printf("\n");
    
    // GPU MMA
    half *d_A, *d_B, *d_C;
    uint32_t *d_meta;
    cudaMalloc(&d_A, 16 * 8 * sizeof(half));
    cudaMalloc(&d_meta, 16 * sizeof(uint32_t));
    cudaMalloc(&d_B, 8 * 16 * sizeof(half));
    cudaMalloc(&d_C, 16 * 8 * sizeof(half));
    
    cudaMemcpy(d_A, A_packed.data(), 16 * 8 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_meta, A_meta.data(), 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), 8 * 16 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, 16 * 8 * sizeof(half));
    
    // Launch single warp
    sparse_mma_tile_kernel<<<1, 32>>>(d_A, d_meta, d_B, d_C);
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    std::vector<half> C_gpu(16 * 8);
    cudaMemcpy(C_gpu.data(), d_C, 16 * 8 * sizeof(half), cudaMemcpyDeviceToHost);
    
    printf("GPU MMA output (first few rows):\n");
    for (int m = 0; m < 4; ++m) {
        printf("Row %d: ", m);
        for (int n = 0; n < 8; ++n) {
            printf("%.1f ", __half2float(C_gpu[m * 8 + n]));
        }
        printf("\n");
    }
    printf("\n");
    
    // Compare
    int errors = 0;
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 8; ++n) {
            float ref = C_ref[m * 8 + n];
            float gpu = __half2float(C_gpu[m * 8 + n]);
            if (std::fabs(ref - gpu) > 0.1f) {
                if (errors < 20) {
                    printf("Mismatch at (%d,%d): ref=%.2f gpu=%.2f\n", m, n, ref, gpu);
                }
                errors++;
            }
        }
    }
    
    if (errors == 0) {
        printf("Unit test PASSED!\n");
    } else {
        printf("\nUnit test FAILED with %d errors\n", errors);
    }
    
    cudaFree(d_A);
    cudaFree(d_meta);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
