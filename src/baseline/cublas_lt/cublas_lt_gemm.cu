/***************************************************************************************************
 * cuBLASLt Multi-Precision GEMM Benchmark
 *
 * D = A * B (alpha=1, beta=0, no C input)
 * A: RowMajor (M x K), B: ColMajor (K x N), D: ColMajor (M x N)
 * Layout: TNN
 *
 * Usage:
 *   ./cublas_profiler --list                               List all config keys
 *   ./cublas_profiler --header                             Print CSV header only
 *   ./cublas_profiler <config> M N K [M N K ...]           Benchmark one or more shapes
 **************************************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../common.cuh"

#define CHECK_CUDA(call)                                                              \
    do {                                                                              \
        cudaError_t err = call;                                                       \
        if (err != cudaSuccess) {                                                     \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                        \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

#define CHECK_CUBLAS(call)                                                                \
    do {                                                                                  \
        cublasStatus_t status = call;                                                     \
        if (status != CUBLAS_STATUS_SUCCESS) {                                            \
            std::cerr << "cuBLASLt error in " << __FILE__ << " line " << __LINE__ << ": " \
                      << status << std::endl;                                             \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)

#define TRY_CUBLAS(call)                                                                \
    do {                                                                                \
        cublasStatus_t status = call;                                                   \
        if (status != CUBLAS_STATUS_SUCCESS) {                                          \
            std::cerr << "cuBLASLt unsupported (" << __LINE__ << "): status=" << status \
                      << std::endl;                                                     \
            return false;                                                               \
        }                                                                               \
    } while (0)

// Timing protocol. Defaults mirror autotuner/profile_worker.py (WARMUP_ITERS, NUM_ROUNDS,
// ITERS_PER_ROUND) so cuBLASLt and CUTLASS are measured identically; eval/cublas.py passes the
// live Python values explicitly so the two cannot drift apart.
static int warmup_iters = 5;
static int num_rounds = 5;
static int iters_per_round = 3;

static constexpr int algo_candidates = 8;
static constexpr int max_num_copies = 64;

/// Operand copies needed so a timed loop always reads A and B cold, mirroring
/// profile_worker.py::_num_copies — ceil(3 * L2 / (A+B bytes)), clamped to [1, max_num_copies].
static int num_copies_for(size_t buf_bytes) {
    if (buf_bytes == 0)
        return 1;
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CHECK_CUDA(cudaGetDeviceProperties(&props, device));
    size_t l2 = props.l2CacheSize > 0 ? size_t(props.l2CacheSize) : size_t(50) << 20;
    size_t n = (3 * l2 + buf_bytes - 1) / buf_bytes;
    return int(std::max<size_t>(1, std::min<size_t>(n, max_num_copies)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Type utilities
///////////////////////////////////////////////////////////////////////////////////////////////////

static size_t dtype_size(cudaDataType_t t) {
    switch (t) {
        case CUDA_R_8F_E4M3:
        case CUDA_R_8F_E5M2:
        case CUDA_R_8I:
            return 1;
        case CUDA_R_16F:
        case CUDA_R_16BF:
            return 2;
        case CUDA_R_32F:
        case CUDA_R_32I:
            return 4;
        case CUDA_R_64F:
            return 8;
        default:
            return 0;
    }
}

static const char* dtype_name(cudaDataType_t t) {
    switch (t) {
        case CUDA_R_8F_E4M3:
            return "e4m3";
        case CUDA_R_8F_E5M2:
            return "e5m2";
        case CUDA_R_8I:
            return "s8";
        case CUDA_R_16F:
            return "f16";
        case CUDA_R_16BF:
            return "bf16";
        case CUDA_R_32F:
            return "f32";
        case CUDA_R_32I:
            return "s32";
        case CUDA_R_64F:
            return "f64";
        default:
            return "unknown";
    }
}

static const char* compute_name(cublasComputeType_t t) {
    switch (t) {
        case CUBLAS_COMPUTE_16F:
            return "f16";
        case CUBLAS_COMPUTE_32F:
            return "f32";
        case CUBLAS_COMPUTE_32F_FAST_TF32:
            return "tf32";
        case CUBLAS_COMPUTE_32I:
            return "s32";
        case CUBLAS_COMPUTE_64F:
            return "f64";
        default:
            return "unknown";
    }
}

static bool is_fp8(cudaDataType_t t) { return t == CUDA_R_8F_E4M3 || t == CUDA_R_8F_E5M2; }

///////////////////////////////////////////////////////////////////////////////////////////////////
// Fill dispatch — delegates to common.cuh fill<T, FillMode::RANDOM>
///////////////////////////////////////////////////////////////////////////////////////////////////

static void fill_random(void* ptr, size_t count, cudaDataType_t dtype, uint64_t seed) {
    switch (dtype) {
        case CUDA_R_16F:
            fill<half, FillMode::RANDOM>((half*)ptr, count, seed, -1.0f, 1.0f);
            break;
        case CUDA_R_16BF:
            fill<__nv_bfloat16, FillMode::RANDOM>((__nv_bfloat16*)ptr, count, seed, -1.0f, 1.0f);
            break;
        case CUDA_R_32F:
            fill<float, FillMode::RANDOM>((float*)ptr, count, seed, -1.0f, 1.0f);
            break;
        case CUDA_R_64F:
            fill<double, FillMode::RANDOM>((double*)ptr, count, seed, -1.0f, 1.0f);
            break;
        case CUDA_R_8F_E4M3:
            fill<__nv_fp8_e4m3, FillMode::RANDOM>((__nv_fp8_e4m3*)ptr, count, seed, -1.0f, 1.0f);
            break;
        case CUDA_R_8F_E5M2:
            fill<__nv_fp8_e5m2, FillMode::RANDOM>((__nv_fp8_e5m2*)ptr, count, seed, -1.0f, 1.0f);
            break;
        case CUDA_R_8I:
            fill<int8_t, FillMode::RANDOM>((int8_t*)ptr, count, seed, -128.0f, 127.0f);
            break;
        default:
            break;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Configuration table
///////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmConfig {
    cudaDataType_t typeA, typeB, typeD;
    cublasComputeType_t computeType;
    cudaDataType_t scaleType;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Layouts — A is logically (M,K), B is logically (K,N), D is always ColMajor (M,N). Matches the
// (layout_a, layout_b) convention used throughout the rest of the repo (e.g. autotuner/
// config_space.py LAYOUTS): "T" = RowMajor, "N" = ColMajor. cuBLAS is natively column-major, so a
// RowMajor operand is described as its ColMajor transpose (dims swapped, ld = the shared K extent)
// with the corresponding transA/transB set — same trick CUTLASS itself relies on.
///////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmLayout {
    const char* name;    // "tn", "tt", "nn", "nt" — matches autotuner's LAYOUT_SHORT convention
    cublasOperation_t transA, transB;
};

static const std::map<std::string, GemmLayout> ALL_LAYOUTS = {
    {"tn", {"tn", CUBLAS_OP_T, CUBLAS_OP_N}},  // A RowMajor, B ColMajor (original hardcoded default)
    {"tt", {"tt", CUBLAS_OP_T, CUBLAS_OP_T}},  // A RowMajor, B RowMajor
    {"nn", {"nn", CUBLAS_OP_N, CUBLAS_OP_N}},  // A ColMajor, B ColMajor
    {"nt", {"nt", CUBLAS_OP_N, CUBLAS_OP_T}},  // A ColMajor, B RowMajor
};

static const std::map<std::string, GemmConfig> ALL_CONFIGS = {
    // ---- FP16 inputs ----
    {"fp16_f16_f16", {CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, CUDA_R_16F}},
    {"fp16_f32_f16", {CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"fp16_f32_f32", {CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F}},

    // ---- BF16 inputs ----
    {"bf16_f32_bf16", {CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"bf16_f32_f32", {CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F}},

    // ---- FP8 e4m3 × e4m3 ----
    {"e4m3_e4m3_f32_e4m3",
     {CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e4m3_e4m3_f32_e5m2",
     {CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e4m3_e4m3_f32_f16",
     {CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e4m3_e4m3_f32_bf16",
     {CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e4m3_e4m3_f32_f32",
     {CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F}},

    // ---- FP8 e4m3 × e5m2 ----
    {"e4m3_e5m2_f32_e4m3",
     {CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e4m3_e5m2_f32_e5m2",
     {CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_8F_E5M2, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e4m3_e5m2_f32_f16",
     {CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e4m3_e5m2_f32_bf16",
     {CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e4m3_e5m2_f32_f32",
     {CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F}},

    // ---- FP8 e5m2 × e5m2 ----
    {"e5m2_e5m2_f32_e4m3",
     {CUDA_R_8F_E5M2, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e5m2_e5m2_f32_e5m2",
     {CUDA_R_8F_E5M2, CUDA_R_8F_E5M2, CUDA_R_8F_E5M2, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e5m2_e5m2_f32_f16",
     {CUDA_R_8F_E5M2, CUDA_R_8F_E5M2, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e5m2_e5m2_f32_bf16",
     {CUDA_R_8F_E5M2, CUDA_R_8F_E5M2, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e5m2_e5m2_f32_f32",
     {CUDA_R_8F_E5M2, CUDA_R_8F_E5M2, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F}},

    // ---- FP8 e5m2 × e4m3 ----
    {"e5m2_e4m3_f32_e4m3",
     {CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e5m2_e4m3_f32_e5m2",
     {CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e5m2_e4m3_f32_f16",
     {CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e5m2_e4m3_f32_bf16",
     {CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F}},
    {"e5m2_e4m3_f32_f32",
     {CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F}},

    // ---- Int8 ----
    {"int8", {CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I}},

    // ---- TF32 ----
    {"tf32", {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F}},

    // ---- Double ----
    {"double", {CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F, CUDA_R_64F}},
};

// One row per cuBLASLt heuristic candidate, all profiled with the same protocol. algo is the
// candidate's heuristic rank (0 = cuBLASLt's own top pick), so "best of the first b" for any
// budget b — and the old best_/top1_ pair — are recovered downstream instead of being baked in.
static const char* CSV_HEADER =
    "config,layout,A_type,B_type,compute,D_type,M,N,K,"
    "algo,mean_ms,std_ms,mean_tflops,std_tflops";

///////////////////////////////////////////////////////////////////////////////////////////////////
// cuBLASLt GEMM wrapper
///////////////////////////////////////////////////////////////////////////////////////////////////

struct CublasLtGemm {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t matmulDesc = nullptr;
    cublasLtMatrixLayout_t layoutA = nullptr;
    cublasLtMatrixLayout_t layoutB = nullptr;
    cublasLtMatrixLayout_t layoutC = nullptr;
    cublasLtMatrixLayout_t layoutD = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulHeuristicResult_t heuristics[algo_candidates];
    int numAlgos = 0;
    void* workspace = nullptr;
    void* dC = nullptr;
    size_t workspaceSize = 0;
    float* d_scale_A = nullptr;
    float* d_scale_B = nullptr;
    float* d_scale_D = nullptr;
    float* d_amax_D = nullptr;
    alignas(8) uint8_t alpha_buf[8] = {};
    alignas(8) uint8_t beta_buf[8] = {};

    /// Returns true on success, false if cuBLASLt cannot handle this combo.
    bool init(int M, int N, int K, const GemmConfig& cfg, const GemmLayout& layout) {
        TRY_CUBLAS(cublasLtCreate(&handle));
        TRY_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, cfg.computeType, cfg.scaleType));

        cublasOperation_t transA = layout.transA;
        cublasOperation_t transB = layout.transB;
        TRY_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA,
                                                  sizeof(transA)));
        TRY_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB,
                                                  sizeof(transB)));

        // FP8 requires per-tensor scale pointers (set to 1.0 for benchmarking)
        if (is_fp8(cfg.typeA) || is_fp8(cfg.typeB)) {
            float h_one = 1.0f;
            CHECK_CUDA(cudaMalloc(&d_scale_A, sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_scale_B, sizeof(float)));
            CHECK_CUDA(cudaMemcpy(d_scale_A, &h_one, sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_scale_B, &h_one, sizeof(float), cudaMemcpyHostToDevice));
            TRY_CUBLAS(cublasLtMatmulDescSetAttribute(
                matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_A, sizeof(d_scale_A)));
            TRY_CUBLAS(cublasLtMatmulDescSetAttribute(
                matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_B, sizeof(d_scale_B)));
        }
        if (is_fp8(cfg.typeD)) {
            float h_one = 1.0f;
            float h_zero = 0.0f;
            CHECK_CUDA(cudaMalloc(&d_scale_D, sizeof(float)));
            CHECK_CUDA(cudaMemcpy(d_scale_D, &h_one, sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&d_amax_D, sizeof(float)));
            CHECK_CUDA(cudaMemcpy(d_amax_D, &h_zero, sizeof(float), cudaMemcpyHostToDevice));
            TRY_CUBLAS(cublasLtMatmulDescSetAttribute(
                matmulDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale_D, sizeof(d_scale_D)));
            TRY_CUBLAS(cublasLtMatmulDescSetAttribute(
                matmulDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &d_amax_D, sizeof(d_amax_D)));
        }

        // Matrix layouts — A is logically (M,K), B is logically (K,N). A RowMajor operand is
        // described to cuBLAS (natively column-major) as its ColMajor transpose with transA=T
        // undoing it; a ColMajor operand is described directly with transA/transB=N. Same for B.
        if (transA == CUBLAS_OP_T) {
            // A RowMajor MxK stored as ColMajor KxM, ld=K
            TRY_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, cfg.typeA, K, M, K));
        } else {
            // A ColMajor MxK, ld=M
            TRY_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, cfg.typeA, M, K, M));
        }
        if (transB == CUBLAS_OP_N) {
            // B ColMajor KxN, ld=K
            TRY_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, cfg.typeB, K, N, K));
        } else {
            // B RowMajor KxN stored as ColMajor NxK, ld=N
            TRY_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, cfg.typeB, N, K, N));
        }
        // D: ColMajor MxN, ld=M
        TRY_CUBLAS(cublasLtMatrixLayoutCreate(&layoutD, cfg.typeD, M, N, M));
        // C: same as D, but cuBLASLt doesn't support FP8 for C, so use bf16 instead
        if (is_fp8(cfg.typeD)) {
            TRY_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, M, N, M));
            CHECK_CUDA(cudaMalloc(&dC, size_t(M) * N * sizeof(__nv_bfloat16)));
            CHECK_CUDA(cudaMemset(dC, 0, size_t(M) * N * sizeof(__nv_bfloat16)));
        } else {
            TRY_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, cfg.typeD, M, N, M));
        }

        // Workspace
        workspaceSize = 32ULL * 1024 * 1024;  // 32 MB — more than enough for cuBLASLt heuristic search
        CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

        // Preference: large workspace + exhaustive search
        TRY_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
        TRY_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference,
                                                        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &workspaceSize, sizeof(workspaceSize)));

        // Request multiple algorithm candidates
        numAlgos = 0;
        TRY_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, layoutA, layoutB, layoutC,
                                                  layoutD, preference, algo_candidates, heuristics,
                                                  &numAlgos));
        if (numAlgos == 0) {
            std::cerr << "cuBLASLt: no algorithm found for this combination" << std::endl;
            return false;
        }

        // Alpha = 1, Beta = 0 in the correct scalar type
        memset(alpha_buf, 0, sizeof(alpha_buf));
        memset(beta_buf, 0, sizeof(beta_buf));
        switch (cfg.scaleType) {
            case CUDA_R_16F: {
                half a = __float2half(1.0f);
                memcpy(alpha_buf, &a, sizeof(a));
                break;
            }
            case CUDA_R_32F: {
                float a = 1.0f;
                memcpy(alpha_buf, &a, sizeof(a));
                break;
            }
            case CUDA_R_32I: {
                int32_t a = 1;
                memcpy(alpha_buf, &a, sizeof(a));
                break;
            }
            case CUDA_R_64F: {
                double a = 1.0;
                memcpy(alpha_buf, &a, sizeof(a));
                break;
            }
            default:
                break;
        }

        return true;
    }

    void runAlgo(void* A, void* B, void* D, cudaStream_t stream, int algoIdx) {
        void* C_ptr = dC ? dC : D;
        CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc, alpha_buf, A, layoutA, B, layoutB, beta_buf,
                                    C_ptr, layoutC, D, layoutD, &heuristics[algoIdx].algo,
                                    workspace, workspaceSize, stream));
    }

    void destroy() {
        if (d_scale_A) {
            cudaFree(d_scale_A);
            d_scale_A = nullptr;
        }
        if (d_scale_B) {
            cudaFree(d_scale_B);
            d_scale_B = nullptr;
        }
        if (d_scale_D) {
            cudaFree(d_scale_D);
            d_scale_D = nullptr;
        }
        if (d_amax_D) {
            cudaFree(d_amax_D);
            d_amax_D = nullptr;
        }
        if (dC) {
            cudaFree(dC);
            dC = nullptr;
        }
        if (workspace) {
            cudaFree(workspace);
            workspace = nullptr;
        }
        if (preference) {
            cublasLtMatmulPreferenceDestroy(preference);
            preference = nullptr;
        }
        if (layoutA) {
            cublasLtMatrixLayoutDestroy(layoutA);
            layoutA = nullptr;
        }
        if (layoutB) {
            cublasLtMatrixLayoutDestroy(layoutB);
            layoutB = nullptr;
        }
        if (layoutC) {
            cublasLtMatrixLayoutDestroy(layoutC);
            layoutC = nullptr;
        }
        if (layoutD) {
            cublasLtMatrixLayoutDestroy(layoutD);
            layoutD = nullptr;
        }
        if (matmulDesc) {
            cublasLtMatmulDescDestroy(matmulDesc);
            matmulDesc = nullptr;
        }
        if (handle) {
            cublasLtDestroy(handle);
            handle = nullptr;
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// CSV output helpers
///////////////////////////////////////////////////////////////////////////////////////////////////

static void print_csv_prefix(const std::string& key, const GemmLayout& layout, const GemmConfig& cfg,
                              int M, int N, int K) {
    std::cout << key << "," << layout.name << "," << dtype_name(cfg.typeA) << ","
              << dtype_name(cfg.typeB) << "," << compute_name(cfg.computeType) << ","
              << dtype_name(cfg.typeD) << "," << M << "," << N << "," << K;
}

/// Warmup + timed rounds for one algorithm, rotating through the operand copies so every
/// iteration reads A and B cold. Returns num_rounds per-iteration mean latencies (ms), matching
/// profile_worker.py::benchmark, whose caller reduces them to mean/stdev the same way.
static std::vector<double> profile_algo(CublasLtGemm& gemm, const std::vector<void*>& As,
                                        const std::vector<void*>& Bs, void* D,
                                        cudaStream_t stream, int algoIdx) {
    const int nc = int(As.size());
    int it = 0;

    for (int i = 0; i < warmup_iters; ++i, ++it)
        gemm.runAlgo(As[it % nc], Bs[it % nc], D, stream, algoIdx);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::vector<double> rounds;
    rounds.reserve(num_rounds);
    for (int r = 0; r < num_rounds; ++r) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        for (int i = 0; i < iters_per_round; ++i, ++it)
            gemm.runAlgo(As[it % nc], Bs[it % nc], D, stream, algoIdx);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        rounds.push_back(double(ms) / iters_per_round);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return rounds;
}

static double mean_of(const std::vector<double>& v) {
    double sum = 0;
    for (double x : v)
        sum += x;
    return sum / v.size();
}

static double stdev_of(const std::vector<double>& v) {
    if (v.size() < 2)
        return 0.0;
    double m = mean_of(v), acc = 0;
    for (double x : v)
        acc += (x - m) * (x - m);
    return std::sqrt(acc / (v.size() - 1));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Benchmark
///////////////////////////////////////////////////////////////////////////////////////////////////

void benchmark(const GemmConfig& cfg, const GemmLayout& layout, const std::string& key, int M, int N,
               int K) {
    sleep_ms(50);

    size_t sA = dtype_size(cfg.typeA);
    size_t sB = dtype_size(cfg.typeB);
    size_t sD = dtype_size(cfg.typeD);

    size_t count_A = size_t(M) * K;
    size_t count_B = size_t(K) * N;
    size_t count_D = size_t(M) * N;
    size_t bytes_A = count_A * sA;
    size_t bytes_B = count_B * sB;

    const int num_copies = num_copies_for(bytes_A + bytes_B);

    std::vector<void*> dAs(num_copies, nullptr);
    std::vector<void*> dBs(num_copies, nullptr);
    void* dD = nullptr;
    for (int i = 0; i < num_copies; ++i) {
        CHECK_CUDA(cudaMalloc(&dAs[i], bytes_A));
        CHECK_CUDA(cudaMalloc(&dBs[i], bytes_B));
        fill_random(dAs[i], count_A, cfg.typeA, 2024 + i);
        fill_random(dBs[i], count_B, cfg.typeB, 3024 + i);
    }
    CHECK_CUDA(cudaMalloc(&dD, count_D * sD));
    CHECK_CUDA(cudaMemset(dD, 0, count_D * sD));
    CHECK_CUDA(cudaDeviceSynchronize());

    auto release = [&]() {
        for (int i = 0; i < num_copies; ++i) {
            CHECK_CUDA(cudaFree(dAs[i]));
            CHECK_CUDA(cudaFree(dBs[i]));
        }
        CHECK_CUDA(cudaFree(dD));
    };

    CublasLtGemm gemm;
    if (!gemm.init(M, N, K, cfg, layout)) {
        print_csv_prefix(key, layout, cfg, M, N, K);
        std::cout << ",-1,UNSUPPORTED,UNSUPPORTED,UNSUPPORTED,UNSUPPORTED" << std::endl;
        gemm.destroy();
        release();
        return;
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    double flops = 2.0 * M * N * K;
    for (int algo = 0; algo < gemm.numAlgos; ++algo) {
        // Cool down so one candidate's clock boost does not carry into the next.
        CHECK_CUDA(cudaDeviceSynchronize());
        sleep_ms(300);

        std::vector<double> rounds = profile_algo(gemm, dAs, dBs, dD, stream, algo);
        std::vector<double> tflops;
        tflops.reserve(rounds.size());
        for (double ms : rounds)
            tflops.push_back((flops / 1e12) / (ms / 1000.0));

        print_csv_prefix(key, layout, cfg, M, N, K);
        std::cout << "," << algo << "," << mean_of(rounds) << "," << stdev_of(rounds) << ","
                  << mean_of(tflops) << "," << stdev_of(tflops) << std::endl;
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    gemm.destroy();
    release();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void print_usage(const char* prog) {
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  " << prog << " --list                               List all config keys"
              << std::endl;
    std::cerr << "  " << prog << " --header                             Print CSV header"
              << std::endl;
    std::cerr << "  " << prog
              << " <config> [--layout tn|tt|nn|nt] [--warmup N] [--rounds N]"
              << " [--iters-per-round N] M N K [M N K ...]" << std::endl;
    std::cerr << "        Benchmarks every cuBLASLt heuristic candidate; one CSV row per algo."
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string arg1 = argv[1];

    if (arg1 == "--list") {
        for (const auto& [key, _] : ALL_CONFIGS)
            std::cout << key << std::endl;
        return 0;
    }

    if (arg1 == "--header") {
        std::cout << CSV_HEADER << std::endl;
        return 0;
    }

    auto it = ALL_CONFIGS.find(arg1);
    if (it == ALL_CONFIGS.end()) {
        std::cerr << "Unknown config: " << arg1 << std::endl;
        std::cerr << "Run with --list to see available configs." << std::endl;
        return 1;
    }
    const GemmConfig& cfg = it->second;

    int shape_start = 2;
    GemmLayout layout = ALL_LAYOUTS.at("tn");
    while (shape_start + 1 < argc && std::string(argv[shape_start]).rfind("--", 0) == 0) {
        std::string flag = argv[shape_start];
        std::string value = argv[shape_start + 1];
        if (flag == "--layout") {
            auto lit = ALL_LAYOUTS.find(value);
            if (lit == ALL_LAYOUTS.end()) {
                std::cerr << "Unknown layout: " << value << " (expected tn, tt, nn, or nt)"
                          << std::endl;
                return 1;
            }
            layout = lit->second;
        } else if (flag == "--warmup") {
            warmup_iters = std::atoi(value.c_str());
        } else if (flag == "--rounds") {
            num_rounds = std::atoi(value.c_str());
        } else if (flag == "--iters-per-round") {
            iters_per_round = std::atoi(value.c_str());
        } else {
            std::cerr << "Unknown flag: " << flag << std::endl;
            return 1;
        }
        shape_start += 2;
    }
    if (warmup_iters < 0 || num_rounds < 2 || iters_per_round < 1) {
        std::cerr << "Invalid protocol: warmup >= 0, rounds >= 2, iters-per-round >= 1"
                  << std::endl;
        return 1;
    }

    int shape_args = argc - shape_start;
    if (shape_args < 3 || shape_args % 3 != 0) {
        std::cerr << "Expected one or more M N K triplets after config key." << std::endl;
        return 1;
    }

    for (int i = shape_start; i < argc; i += 3) {
        int M = std::atoi(argv[i]);
        int N = std::atoi(argv[i + 1]);
        int K = std::atoi(argv[i + 2]);
        benchmark(cfg, layout, arg1, M, N, K);
    }

    return 0;
}
