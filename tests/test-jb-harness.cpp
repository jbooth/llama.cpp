#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float * gen_rand_f32(int64_t n) {
    float * data = (float *) malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; ++i) {
        data[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    }
    return data;
}

static void matmul_ref(float *C, const float *A, const float *B, int64_t M, int64_t N, int64_t K) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < K; ++j) {
            float acc = 0.0f;
            for (int64_t k = 0; k < N; ++k) {
                acc += A[i*N + k] * B[k*K + j];
            }
            C[i*K + j] = acc;
        }
    }
}

static void compare_f32(const float * ref, const float * out, int64_t n, float * max_err, float * rms_err) {
    *max_err = 0.0f;
    double sum_sq_err = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        float err = fabsf(ref[i] - out[i]);
        if (err > *max_err) *max_err = err;
        sum_sq_err += (double)err * err;
    }
    *rms_err = sqrt(sum_sq_err / n);
}

// Converts Row-Major (src_rows x src_cols) to GGML Layout (ne0=src_cols, ne1=src_rows)
static void fill_tensor(struct ggml_tensor * t, const float * src, int64_t src_rows, int64_t src_cols, ggml_type qtype) {
    // In GGML: ne0 is the "width" (elements per row).
    // To match a row-major src, ne0 must be src_cols.
    GGML_ASSERT(t->ne[0] == src_cols);
    GGML_ASSERT(t->ne[1] == src_rows);

    float * tmp_f32 = (float *) malloc(src_rows * src_cols * sizeof(float));

    // Transfer data: This is essentially a transpose if you consider
    // how GGML views memory vs standard C row-major.
    for (int64_t r = 0; r < src_rows; ++r) {
        for (int64_t c = 0; c < src_cols; ++c) {
            // GGML stores rows contiguously.
            // So for row 'r', the index of element 'c' is r * ne0 + c.
            tmp_f32[r * src_cols + c] = src[r * src_cols + c];
        }
    }

    if (qtype == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, tmp_f32, 0, ggml_nbytes(t));
    } else {
        ggml_quantize_init(qtype);
        void * tmp_q = malloc(ggml_nbytes(t));
        // Quantize src_rows (ne1) each of length src_cols (ne0)
        ggml_quantize_chunk(qtype, tmp_f32, tmp_q, 0, t->ne[1], t->ne[0], NULL);
        ggml_backend_tensor_set(t, tmp_q, 0, ggml_nbytes(t));
        free(tmp_q);
    }
    free(tmp_f32);
}

void test_matmul(ggml_backend_t backend, int64_t M, int64_t N, int64_t K, ggml_type quant_type) {
    srand(0x1234);

    float * A_ref = gen_rand_f32(M * N);
    float * B_ref = gen_rand_f32(N * K);
    float * C_ref = (float *) malloc(M * K * sizeof(float));
    float * C_out = (float *) malloc(M * K * sizeof(float));

    matmul_ref(C_ref, A_ref, B_ref, M, N, K);

    struct ggml_init_params ip = { .mem_size = 1024*1024*1024, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(ip);

    // LOGIC:
    // Reference: C(M,K) = A(M,N) * B(N,K)
    // GGML mul_mat(t0, t1) computes t1 * t0^T.
    // So we want: A(M,N) * Bq(K,N)^T
    // t1 (A)  shape: ne0=N, ne1=M  (Transposed view of A_ref)
    // t0 (Bq) shape: ne0=N, ne1=K  (Transposed view of B_ref)

    struct ggml_tensor * A  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    struct ggml_tensor * Bq = ggml_new_tensor_2d(ctx, quant_type,   N, K);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * C = ggml_mul_mat_tiled(ctx, Bq, A);
    ggml_build_forward_expand(gf, C);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    // A_ref is M rows, N cols. GGML A is N wide, M high.
    fill_tensor(A, A_ref, M, N, GGML_TYPE_F32);

    // B_ref is N rows, K cols.
    // To make GGML Bq (N wide, K high), we must treat B_ref as K rows of N.
    // We effectively transpose B_ref here.
    float * B_ref_T = (float *) malloc(N * K * sizeof(float));
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            B_ref_T[k * N + n] = B_ref[n * K + k];
        }
    }
    fill_tensor(Bq, B_ref_T, K, N, quant_type);

    ggml_backend_graph_compute(backend, gf);

    // C is (K, M). Result C_ref is (M, K). Read rows of C into C_out.
    for (int64_t i = 0; i < M; ++i) {
        ggml_backend_tensor_get(C, C_out + i*K, i * C->nb[1], K * sizeof(float));
    }

    float max_err, rms_err;
    compare_f32(C_ref, C_out, M*K, &max_err, &rms_err);
    float tol = (quant_type == GGML_TYPE_F32) ? 1e-4f : 0.5f;

    printf("TEST %lldx%lld * %lldx%lld (%s): %s (max_err: %f)\n",
           (long long)M, (long long)N, (long long)N, (long long)K,
           ggml_type_name(quant_type), (max_err <= tol) ? "PASS" : "FAIL", max_err);

    ggml_free(ctx);
    free(A_ref); free(B_ref); free(B_ref_T); free(C_ref); free(C_out);
}

void bench_matmul(ggml_backend_t backend, int64_t dim, ggml_type quant_type) {
    int64_t M = dim, N = dim, K = dim;
    int num_iterations = 10;

    // 1. Prepare data
    float * A_data = gen_rand_f32(M * N);
    float ** B_datas = (float **) malloc(num_iterations * sizeof(float *));
    for (int i = 0; i < num_iterations; ++i) {
        B_datas[i] = gen_rand_f32(N * K);
    }

    // 2. GGML Setup
    struct ggml_init_params ip = { .mem_size = 1024*1024*1024, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * A  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    struct ggml_tensor * Bq = ggml_new_tensor_2d(ctx, quant_type,   N, K);
    struct ggml_tensor * C  = ggml_mul_mat(ctx, Bq, A);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);

    // Allocate on backend
    ggml_backend_alloc_ctx_tensors(ctx, backend);

    // 3. Pre-upload A and pre-quantize all B matrices to save time
    fill_tensor(A, A_data, M, N, GGML_TYPE_F32);

    struct ggml_tensor ** B_tensors = (ggml_tensor**) malloc(num_iterations * sizeof(struct ggml_tensor *));
    for (int i = 0; i < num_iterations; ++i) {
        // We create temporary tensors just to hold the quantized data on the backend
        B_tensors[i] = ggml_new_tensor_2d(ctx, quant_type, N, K);
        ggml_backend_alloc_ctx_tensors(ctx, backend);


        // Transpose and Fill
        float * B_ref_T = (float *) malloc(N * K * sizeof(float));
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) B_ref_T[k * N + n] = B_datas[i][n * K + k];
        }
        fill_tensor(B_tensors[i], B_ref_T, K, N, quant_type);
        free(B_ref_T);
    }

    printf("Benchmarking %s: %lldx%lld matmul (%d iterations)...\n",
           ggml_type_name(quant_type), (long long)dim, (long long)dim, num_iterations);

    // 4. Execution Loop
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < num_iterations; ++i) {
        // Point the graph's Bq to the pre-quantized buffer
        // Note: In a real scenario, you'd usually swap data or use a weight-sharded approach
        // Here we use ggml_backend_tensor_copy for a clean swap into the compute graph tensor
        ggml_backend_tensor_copy(B_tensors[i], Bq);

        ggml_backend_graph_compute(backend, gf);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    // 5. Calculate Results
    double seconds = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double tflops = (2.0 * M * N * K * num_iterations) / (seconds * 1e12);

    printf("  Total time: %.4f s\n", seconds);
    printf("  Avg time:   %.4f s/iter\n", seconds / num_iterations);
    printf("  Throughput: %.4f TFLOPS\n\n", tflops);

    // Cleanup
    for (int i = 0; i < num_iterations; ++i) free(B_datas[i]);
    free(B_datas); free(B_tensors); free(A_data);
    ggml_free(ctx);
}

void bench_matmul_comparison(ggml_backend_t backend, int64_t dim, ggml_type quant_type) {
    int64_t M = dim, N = dim, K = dim;
    int num_iterations = 10;

    // 1. Prepare data
    float * A_data = gen_rand_f32(M * N);
    float ** B_datas = (float **) malloc(num_iterations * sizeof(float *));
    for (int i = 0; i < num_iterations; ++i) {
        B_datas[i] = gen_rand_f32(N * K);
    }

    struct ggml_init_params ip = { .mem_size = 512*1024*1024, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * A  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    struct ggml_tensor * Bq = ggml_new_tensor_2d(ctx, quant_type,   N, K);

    // 2. Build Graphs
    struct ggml_cgraph * gf_std = ggml_new_graph(ctx);
    struct ggml_tensor * C_std  = ggml_mul_mat(ctx, Bq, A);
    ggml_build_forward_expand(gf_std, C_std);

    struct ggml_cgraph * gf_tiled = ggml_new_graph(ctx);
    struct ggml_tensor * C_tiled  = ggml_mul_mat_tiled(ctx, Bq, A);
    ggml_build_forward_expand(gf_tiled, C_tiled);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    // 3. Setup Tensors
    fill_tensor(A, A_data, M, N, GGML_TYPE_F32);
    struct ggml_tensor ** B_tensors = (ggml_tensor **) malloc(num_iterations * sizeof(struct ggml_tensor *));
    for (int i = 0; i < num_iterations; ++i) {
        B_tensors[i] = ggml_new_tensor_2d(ctx, quant_type, N, K);
        ggml_backend_alloc_ctx_tensors(ctx, backend);

        float * B_ref_T = (float *) malloc(N * K * sizeof(float));
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) B_ref_T[k * N + n] = B_datas[i][n * K + k];
        }
        fill_tensor(B_tensors[i], B_ref_T, K, N, quant_type);
        free(B_ref_T);
    }

    // --- INTEGRITY CHECK PHASE ---
    printf("Validating Tiled Output Accuracy...\n");
    ggml_backend_tensor_copy(B_tensors[0], Bq);
    ggml_backend_graph_compute(backend, gf_std);
    ggml_backend_graph_compute(backend, gf_tiled);

    float * out_std   = (float *) malloc(M * K * sizeof(float));
    float * out_tiled = (float *) malloc(M * K * sizeof(float));

    // Get results from backend
    ggml_backend_tensor_get(C_std,   out_std,   0, ggml_nbytes(C_std));
    ggml_backend_tensor_get(C_tiled, out_tiled, 0, ggml_nbytes(C_tiled));

    float max_err, rms_err;
    compare_f32(out_std, out_tiled, M * K, &max_err, &rms_err);

    // Standard GGML kernels and custom tiled kernels should be identical for F32,
    // and extremely close (epsilon differences) for quantized types due to accumulation order.
    bool pass = (max_err < 1e-4f);
    printf("  Integrity: %s (Max Err: %f, RMS Err: %f)\n", pass ? "PASS" : "FAIL", max_err, rms_err);

    if (!pass) {
        printf("  WARNING: Tiled results deviate from standard implementation!\n");
    }

    // --- PERFORMANCE BENCHMARK PHASE ---
    printf("Benchmarking %lldx%lld (%s)...\n", (long long)dim, (long long)dim, ggml_type_name(quant_type));

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < num_iterations; ++i) {
        ggml_backend_tensor_copy(B_tensors[i], Bq);
        ggml_backend_graph_compute(backend, gf_std);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_std = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < num_iterations; ++i) {
        ggml_backend_tensor_copy(B_tensors[i], Bq);
        ggml_backend_graph_compute(backend, gf_tiled);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_tiled = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("  Standard: %.4f s\n", time_std);
    printf("  Tiled:    %.4f s\n", time_tiled);
    printf("  Speedup:  %.2fx\n\n", time_std / time_tiled);

    // Cleanup
    free(out_std); free(out_tiled);
    for (int i = 0; i < num_iterations; ++i) free(B_datas[i]);
    free(B_datas); free(B_tensors); free(A_data);
    ggml_free(ctx);
}

int main(void) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_Q6_K);
    test_matmul(backend, 18, 1024, 7, GGML_TYPE_Q5_K);

    bench_matmul_comparison(backend, 8192, GGML_TYPE_F32);
    bench_matmul_comparison(backend, 8192, GGML_TYPE_Q8_0);
    bench_matmul_comparison(backend, 8192, GGML_TYPE_Q4_K);
    ggml_backend_free(backend);
    return 0;
}