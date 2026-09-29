/*
 * ncu_launcher.cu — Minimal CUDA launcher for NCU profiling.
 *
 * Replaces the Python ncu_worker.py __main__ path. Eliminates ~2 s of
 * Python + torch import overhead per (kernel, shape) pair.
 *
 * Invoked by ncu exactly as the Python script was:
 *
 *   ncu --metrics ... --kernel-name device_kernel --launch-count 1 \
 *       ./build_cache/ncu_launcher                                  \
 *       --build-dir <dir> --so <file> --kernel-id <id>              \
 *       --dtype-a bf16 --dtype-b bf16 --dtype-c bf16                \
 *       --device-id <id> --M <M> --N <N> --K <K>
 *
 * Build (once per node, handled by scheduler.py on first NCU run):
 *
 *   nvcc -O2 -o <build_dir>/ncu_launcher autotuner/ncu_launcher.cu \
 *        -ldl -lcuda
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

typedef int (*run_kernel_once_fn)(int, void**, void**, void*, int, int, int, int);

static int dtype_bytes(const char* dtype) {
    if (!strcmp(dtype, "bf16") || !strcmp(dtype, "f16")) return 2;
    if (!strcmp(dtype, "f32") || !strcmp(dtype, "tf32") || !strcmp(dtype, "s32")) return 4;
    if (!strcmp(dtype, "f64")) return 8;
    return 1; // e4m3, e5m2, s8
}

static const char* get_arg(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc - 1; i++)
        if (!strcmp(argv[i], flag)) return argv[i + 1];
    fprintf(stderr, "Missing argument: %s\n", flag);
    exit(1);
}

int main(int argc, char** argv) {
    const char* build_dir = get_arg(argc, argv, "--build-dir");
    const char* so_file   = get_arg(argc, argv, "--so");
    int kernel_id         = atoi(get_arg(argc, argv, "--kernel-id"));
    int device_id         = atoi(get_arg(argc, argv, "--device-id"));
    int M                 = atoi(get_arg(argc, argv, "--M"));
    int N                 = atoi(get_arg(argc, argv, "--N"));
    int K                 = atoi(get_arg(argc, argv, "--K"));
    int bytes_a           = dtype_bytes(get_arg(argc, argv, "--dtype-a"));
    int bytes_b           = dtype_bytes(get_arg(argc, argv, "--dtype-b"));
    int bytes_c           = dtype_bytes(get_arg(argc, argv, "--dtype-c"));

    // Construct full path to the .so
    char so_path[4096];
    snprintf(so_path, sizeof(so_path), "%s/%s", build_dir, so_file);

    CUDA_CHECK(cudaSetDevice(device_id));

    void* handle = dlopen(so_path, RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    run_kernel_once_fn run_kernel_once =
        (run_kernel_once_fn)dlsym(handle, "run_kernel_once");
    if (!run_kernel_once) {
        fprintf(stderr, "dlsym(run_kernel_once) failed: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }

    void *d_A, *d_B, *d_D;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * K * bytes_a));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)K * N * bytes_b));
    CUDA_CHECK(cudaMalloc(&d_D, (size_t)M * N * bytes_c));
    CUDA_CHECK(cudaMemset(d_A, 0, (size_t)M * K * bytes_a));
    CUDA_CHECK(cudaMemset(d_B, 0, (size_t)K * N * bytes_b));

    void* ptrs_A[1] = {d_A};
    void* ptrs_B[1] = {d_B};

    // Single launch — ncu intercepts and replays internally
    int status = run_kernel_once(kernel_id, ptrs_A, ptrs_B, d_D, 1, M, N, K);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
    dlclose(handle);
    return status;
}
