// Minimal perf driver for the tiled q4_K 4096^3 matmul (M3 target).
// One graph, compute in a loop; pin with taskset, wrap with perf record.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void fill_tensor(struct ggml_tensor * t, const float * src, int64_t src_rows, int64_t src_cols, ggml_type qtype) {
    float * tmp_f32 = (float *) malloc(src_rows * src_cols * sizeof(float));
    for (int64_t r = 0; r < src_rows; ++r)
        for (int64_t c = 0; c < src_cols; ++c)
            tmp_f32[r * src_cols + c] = src[r * src_cols + c];

    if (qtype == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, tmp_f32, 0, ggml_nbytes(t));
    } else {
        ggml_quantize_init(qtype);
        void * tmp_q = malloc(ggml_nbytes(t));
        ggml_quantize_chunk(qtype, tmp_f32, tmp_q, 0, t->ne[1], t->ne[0], NULL);
        ggml_backend_tensor_set(t, tmp_q, 0, ggml_nbytes(t));
        free(tmp_q);
    }
    free(tmp_f32);
}

int main(int argc, char ** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 200;
    srand(0x1234);

    const int64_t M = 4096, N = 4096, K = 4096;

    float * src1_ref = (float *) malloc(M * N * sizeof(float));
    float * src0_ref = (float *) malloc(N * K * sizeof(float));
    for (int64_t i = 0; i < M * N; ++i) src1_ref[i] = ((float)rand() / RAND_MAX - 0.5f) * 5.0f;
    for (int64_t i = 0; i < N * K; ++i) src0_ref[i] = ((float)rand() / RAND_MAX - 0.5f) * 5.0f;

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 8);

    struct ggml_init_params ip = { .mem_size = 1024 * 1024 * 1024, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(ip);

    // dst(M,K) = src1(M,N) x src0(N,K) ; src0 = weights (K rows of N), src1 = activations
    struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    struct ggml_tensor * src0q = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, N, K);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * dst = ggml_mul_mat_tiled(ctx, src0q, src1);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_alloc_ctx_tensors(ctx, backend);
    fill_tensor(src1, src1_ref, M, N, GGML_TYPE_F32);
    fill_tensor(src0q, src0_ref, N, K, GGML_TYPE_Q4_K);

    // warmup (first run pays TLS allocs)
    ggml_backend_graph_compute(backend, gf);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    const double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    const double flops = 2.0 * M * N * K;
    printf("q4_K %lld^3 x %d iters: %.3f s, %.2f TFLOPS\n",
           (long long) M, iters, sec, flops * iters / sec / 1e12);

    return 0;
}