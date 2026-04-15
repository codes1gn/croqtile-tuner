// iter002_warp2x4.cu — Pure CUDA WMMA matmul, bf16->fp32
// Shape: M=16384, N=16384, K=512
// A[M,K] row-major (bf16), B[K,N] row-major (bf16), C[M,N] row-major (fp32)
//
// Strategy: 128x128 output tile, 4x2 warp grid (8 warps, 256 threads).
// Each warp owns a 32x64 region (2x4 WMMA(16x16) tiles) to increase
// per-warp tensor-core work and reduce scheduler overhead.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

using namespace nvcuda;

#define CHECK_CUDA(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
} while(0)

static const int M_SZ=16384, N_SZ=16384, K_SZ=512;
static const int WARMUP=10, ITERS=50, SAMPLES=5;

static const int BM=128, BN=128, BK=16;
static const int WMMA_M=16, WMMA_N=16, WMMA_K=16;
static const int WARPS_M=4, WARPS_N=2, WARP_SIZE=32;
static const int BLOCK_THREADS=WARPS_M*WARPS_N*WARP_SIZE;  // 256
static const int WM_TILES=BM/WARPS_M/WMMA_M;   // 2
static const int WN_TILES=BN/WARPS_N/WMMA_N;   // 4

// Smem: pad by 8 to avoid bank conflicts (each element is 2 bytes)
static const int A_LD=BK+8;         // 24
static const int B_LD=BN+8;         // 136
static const int SMEM_A_SZ=BM*A_LD;  // 3072 per stage
static const int SMEM_B_SZ=BK*B_LD;  // 2176 per stage
static const int STAGES=2;

__global__ __launch_bounds__(BLOCK_THREADS, 2)
void matmul_bf16_fp32_warp2x4(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float*               __restrict__ C,
    int M, int N, int K)
{
    __shared__ __nv_bfloat16 smA[STAGES][BM][A_LD];
    __shared__ __nv_bfloat16 smB[STAGES][BK][B_LD];

    int tile_m=blockIdx.y, tile_n=blockIdx.x;
    int tid=threadIdx.x;
    int warp_id=tid/WARP_SIZE;
    int warp_m=warp_id/WARPS_N;   // 0..3
    int warp_n=warp_id%WARPS_N;   // 0..1

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WM_TILES][WN_TILES];
    for(int i=0;i<WM_TILES;i++)
        for(int j=0;j<WN_TILES;j++)
            wmma::fill_fragment(acc[i][j], 0.f);

    int gm0=tile_m*BM, gn0=tile_n*BN;
    int num_k=K/BK;

    auto load_tile = [&](int s, int kt) {
        int gk0 = kt * BK;
        for(int e=tid; e<BM*BK; e+=BLOCK_THREADS) {
            int r=e/BK, c=e%BK;
            int gm=gm0+r, gk=gk0+c;
            smA[s][r][c] = (gm<M && gk<K) ? A[gm*K+gk] : (__nv_bfloat16)0.f;
        }
        for(int e=tid; e<BK*BN; e+=BLOCK_THREADS) {
            int r=e/BN, c=e%BN;
            int gk=gk0+r, gn=gn0+c;
            smB[s][r][c] = (gk<K && gn<N) ? B[gk*N+gn] : (__nv_bfloat16)0.f;
        }
    };

    load_tile(0, 0);
    __syncthreads();

    for(int k=0; k<num_k; k++) {
        int cur = k & 1;
        int nxt = 1 - cur;

        int wm_off = warp_m * (BM/WARPS_M);   // 0,32,64,96
        int wn_off = warp_n * (BN/WARPS_N);   // 0,64

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       __nv_bfloat16, wmma::row_major> fA[WM_TILES];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       __nv_bfloat16, wmma::row_major> fB[WN_TILES];

        for(int i=0;i<WM_TILES;i++)
            wmma::load_matrix_sync(fA[i],
                (const __nv_bfloat16*)smA[cur] + (wm_off+i*WMMA_M)*A_LD,
                A_LD);
        for(int j=0;j<WN_TILES;j++)
            wmma::load_matrix_sync(fB[j],
                (const __nv_bfloat16*)smB[cur] + wn_off+j*WMMA_N,
                B_LD);
        for(int i=0;i<WM_TILES;i++)
            for(int j=0;j<WN_TILES;j++)
                wmma::mma_sync(acc[i][j], fA[i], fB[j], acc[i][j]);

        __syncthreads();
        if(k+1 < num_k) {
            load_tile(nxt, k+1);
        }
        __syncthreads();
    }

    int gm_w=gm0+warp_m*(BM/WARPS_M);
    int gn_w=gn0+warp_n*(BN/WARPS_N);
    for(int i=0;i<WM_TILES;i++)
        for(int j=0;j<WN_TILES;j++){
            int r=gm_w+i*WMMA_M, c=gn_w+j*WMMA_N;
            if(r<M && c<N)
                wmma::store_matrix_sync(C+r*N+c, acc[i][j], N, wmma::mem_row_major);
        }
}

static void cpu_ref(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                    int m, int n, int K_full, int N_full) {
    for(int i=0;i<m;i++) for(int j=0;j<n;j++){
        float s=0; for(int kk=0;kk<K_full;kk++)
            s+=__bfloat162float(A[i*K_full+kk])*__bfloat162float(B[kk*N_full+j]);
        C[i*n+j]=s;
    }
}

int main(){
    const int M=M_SZ, N=N_SZ, K=K_SZ;
    __nv_bfloat16 *dA,*dB; float *dC;
    CHECK_CUDA(cudaMalloc(&dA,(size_t)M*K*sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dB,(size_t)K*N*sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC,(size_t)M*N*sizeof(float)));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f,1.f);
    size_t szA=(size_t)M*K, szB=(size_t)K*N;
    __nv_bfloat16 *hA=new __nv_bfloat16[szA], *hB=new __nv_bfloat16[szB];
    for(size_t i=0;i<szA;i++) hA[i]=__float2bfloat16(dist(rng));
    for(size_t i=0;i<szB;i++) hB[i]=__float2bfloat16(dist(rng));
    CHECK_CUDA(cudaMemcpy(dA,hA,szA*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,hB,szB*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));

    dim3 block(BLOCK_THREADS);
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);

    {
        CHECK_CUDA(cudaMemset(dC,0,(size_t)M*N*sizeof(float)));
        matmul_bf16_fp32_warp2x4<<<grid,block>>>(dA,dB,dC,M,N,K);
        CHECK_CUDA(cudaDeviceSynchronize());
        const int VM=16,VN=16;
        float hC[VM*VN], refC[VM*VN];
        CHECK_CUDA(cudaMemcpy2D(hC,VN*sizeof(float),dC,N*sizeof(float),
                                VN*sizeof(float),VM,cudaMemcpyDeviceToHost));
        cpu_ref(hA,hB,refC,VM,VN,K,N);
        float max_rel=0;
        for(int i=0;i<VM*VN;i++){
            float denom=fmaxf(fabsf(refC[i]),1e-3f);
            max_rel=fmaxf(max_rel,fabsf(hC[i]-refC[i])/denom);
        }
        if(max_rel<5e-2f) printf("VERIFY: PASS max_rel_err=%.5f\n",max_rel);
        else               printf("VERIFY: FAIL max_rel_err=%.5f\n",max_rel);
    }

    for(int i=0;i<WARMUP;i++)
        matmul_bf16_fp32_warp2x4<<<grid,block>>>(dA,dB,dC,M,N,K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t t0,t1;
    CHECK_CUDA(cudaEventCreate(&t0)); CHECK_CUDA(cudaEventCreate(&t1));
    double tsum=0;
    for(int s=0;s<SAMPLES;s++){
        CHECK_CUDA(cudaEventRecord(t0));
        for(int i=0;i<ITERS;i++)
            matmul_bf16_fp32_warp2x4<<<grid,block>>>(dA,dB,dC,M,N,K);
        CHECK_CUDA(cudaEventRecord(t1)); CHECK_CUDA(cudaEventSynchronize(t1));
        float ms; CHECK_CUDA(cudaEventElapsedTime(&ms,t0,t1)); ms/=ITERS;
        double tf=2.0*M*N*K/ms/1e9; tsum+=tf;
        printf("sample %d: time=%.3f ms, tflops=%.2f\n",s+1,ms,tf);
    }
    double avg=tsum/SAMPLES;
    printf("\nTFLOPS: %.2f   time_ms: %.3f\n",avg,2.0*M*N*K/avg/1e9);
    CHECK_CUDA(cudaEventDestroy(t0)); CHECK_CUDA(cudaEventDestroy(t1));
    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
    delete[] hA; delete[] hB;
    return 0;
}
