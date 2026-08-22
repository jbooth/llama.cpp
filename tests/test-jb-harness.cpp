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
        data[i] *= 5.0f;
    }
    return data;
}

static void compare_f32(const float * ref, const float * out, int64_t n, float * max_err, float * rms_err) {
    *max_err = 0.0f;
    double sum_sq_err = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        float err = fabsf(ref[i] - out[i]);
        if (err > *max_err) {
            printf("New max err %f  ref %f out %f \n", *max_err, ref[i], out[i]);
            *max_err = err;
        }
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
    float * C_out = (float *) malloc(M * K * sizeof(float));
    float * C_tiled = (float *) malloc(M * K * sizeof(float));

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

    struct ggml_cgraph * gf  = ggml_new_graph(ctx);
    struct ggml_tensor * C   = ggml_mul_mat(ctx, Bq, A);
    ggml_build_forward_expand(gf, C);

    struct ggml_cgraph * gf_t  = ggml_new_graph(ctx);
    struct ggml_tensor * C_t   = ggml_mul_mat_tiled(ctx, Bq, A);
    ggml_build_forward_expand(gf_t, C_t);

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
    ggml_backend_graph_compute(backend, gf_t);

    // C is (K, M). Result C_ref is (M, K). Read rows of C into C_out.
    for (int64_t i = 0; i < M; ++i) {
        ggml_backend_tensor_get(C,   C_out + i*K,   i * C->nb[1], K * sizeof(float));
        ggml_backend_tensor_get(C_t, C_tiled + i*K,   i * C_t->nb[1], K * sizeof(float));
    }

    // max |C|: used as scale for quantization tolerance
    float scale = 0.0f;
    for (int64_t i = 0; i < M*K; ++i) {
        scale = fmaxf(scale, fabsf(C_out[i]));
    }

    // std vs tiled: identical quantized inputs, so any large difference here is a bug in the tiled kernel
    float max_err, rms_err;
    compare_f32(C_out, C_tiled, M*K, &max_err, &rms_err);
    float tol = (quant_type == GGML_TYPE_F32) ? 1e-4f : fmaxf(1e-3f, 1e-3f * scale);

    printf("TEST %lldx%lld * %lldx%lld (%s): %s (max_err: %f, rms: %f, scale: %f)\n",
           (long long)M, (long long)N, (long long)N, (long long)K,
           ggml_type_name(quant_type), (max_err <= tol) ? "PASS" : "FAIL", max_err, rms_err, scale);

    // if the tiled kernel deviates from std beyond quantization tolerance, dump a few offenders
    if (max_err > tol) {
        int64_t shown = 0;
        for (int64_t i = 0; i < M*K && shown < 8; ++i) {
            float err = fabsf(C_out[i] - C_tiled[i]);
            if (err > tol) {
                printf("  tiled vs std: i=%lld (m=%lld k=%lld) std=%f tiled=%f err=%f\n",
                       (long long)i, (long long)(i/K), (long long)(i%K), C_out[i], C_tiled[i], err);
                ++shown;
            }
        }
    }

    ggml_free(ctx);
    free(A_ref); free(B_ref); free(B_ref_T); free(C_out); free(C_tiled);
    //if (max_err > tol) { exit(1); }
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

static double time_graph_compute(ggml_backend_t backend, struct ggml_cgraph * gf) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ggml_backend_graph_compute(backend, gf);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
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

// Time 10 matmuls of the given dimensions with a fresh random A and B each
// iteration, comparing ggml_mul_mat (std) against ggml_mul_mat_tiled.
// Fresh inputs per iteration avoid favorable cache states from reusing the
// same matrices; the start order is alternated to avoid warm-up bias.
void bench_tiled_vs_std(ggml_backend_t backend, int64_t M, int64_t N, int64_t K, ggml_type quant_type) {
    const int num_iterations = 10;

    struct ggml_init_params ip = { .mem_size = 512*1024*1024, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * A  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    struct ggml_tensor * Bq = ggml_new_tensor_2d(ctx, quant_type,   N, K);

    struct ggml_cgraph * gf_std  = ggml_new_graph(ctx);
    struct ggml_tensor * C_std   = ggml_mul_mat(ctx, Bq, A);
    ggml_build_forward_expand(gf_std, C_std);

    struct ggml_cgraph * gf_tiled = ggml_new_graph(ctx);
    struct ggml_tensor * C_tiled  = ggml_mul_mat_tiled(ctx, Bq, A);
    ggml_build_forward_expand(gf_tiled, C_tiled);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    double time_std = 0.0, time_tiled = 0.0;

    for (int i = 0; i < num_iterations; ++i) {
        // fresh random data every iteration
        float * A_data = gen_rand_f32(M * N);
        float * B_data = gen_rand_f32(N * K);

        // Bq is K rows of N in ggml layout, so transpose B into it
        float * B_T = (float *) malloc(N * K * sizeof(float));
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < K; ++k) {
                B_T[k * N + n] = B_data[n * K + k];
            }
        }
        fill_tensor(A,  A_data, M, N, GGML_TYPE_F32);
        fill_tensor(Bq, B_T, K, N, quant_type);
        free(A_data); free(B_data); free(B_T);

        if (i % 2 == 0) {
            time_std   += time_graph_compute(backend, gf_std);
            time_tiled += time_graph_compute(backend, gf_tiled);
        } else {
            time_tiled += time_graph_compute(backend, gf_tiled);
            time_std   += time_graph_compute(backend, gf_std);
        }
    }

    const double tflops_std   = (2.0 * M * N * K * num_iterations) / (time_std   * 1e12);
    const double tflops_tiled = (2.0 * M * N * K * num_iterations) / (time_tiled * 1e12);

    printf("BENCH %lldx%lld * %lldx%lld (%s), %d iters: std %.4f s (%.3f TFLOPS), tiled %.4f s (%.3f TFLOPS), speedup %.2fx\n",
           (long long)M, (long long)N, (long long)N, (long long)K,
           ggml_type_name(quant_type), num_iterations,
           time_std, tflops_std, time_tiled, tflops_tiled, time_std / time_tiled);

    ggml_free(ctx);
}

// Fetch the CPU_REPACK extra buffer type through the public proc-address API,
// the same way the model loader does. Returns NULL if repack is not built in.
static ggml_backend_buffer_type_t get_cpu_repack_buft(void) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        return NULL;
    }
    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    ggml_backend_dev_get_extra_bufts_t get_extra =
        (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
    if (!get_extra) {
        return NULL;
    }
    ggml_backend_buffer_type_t * bufts = get_extra(cpu_dev);
    // the only extra buffer type the CPU backend exposes is CPU_REPACK
    return (bufts && *bufts) ? *bufts : NULL;
}

// Three-way benchmark: standard ggml_mul_mat, repacked ggml_mul_mat, and the
// custom tiled kernel. std and tiled share a Q4_K weight tensor in the default
// CPU buffer; repack uses a second Q4_K tensor allocated in the CPU_REPACK
// buffer (whose set_tensor repacks the raw quants, and whose compute kernel is
// selected automatically because the weight lives in that buffer type).
void bench_three_way(ggml_backend_t backend, int64_t M, int64_t N, int64_t K, ggml_type quant_type) {
    const int num_iterations = 10;

    ggml_backend_buffer_type_t repack_buft = get_cpu_repack_buft();
    if (!repack_buft) {
        printf("BENCH %lldx%lld * %lldx%lld (%s): CPU_REPACK buffer type unavailable, skipping repack path\n",
               (long long)M, (long long)N, (long long)N, (long long)K, ggml_type_name(quant_type));
        return;
    }
    // repack 8x8 layout requires K (ne1) and N (ne0) to be multiples of 8
    if (K % 8 != 0 || N % 8 != 0) {
        printf("BENCH %lldx%lld * %lldx%lld (%s): N/K not multiples of 8, skipping repack path\n",
               (long long)M, (long long)N, (long long)N, (long long)K, ggml_type_name(quant_type));
        return;
    }

    struct ggml_init_params ip = { .mem_size = 1024*1024*1024, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * A      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    struct ggml_tensor * Bq_std = ggml_new_tensor_2d(ctx, quant_type,   N, K);
    struct ggml_tensor * Bq_rep = ggml_new_tensor_2d(ctx, quant_type,   N, K);

    struct ggml_cgraph * gf_std   = ggml_new_graph(ctx);
    struct ggml_tensor * C_std    = ggml_mul_mat(ctx, Bq_std, A);
    ggml_build_forward_expand(gf_std, C_std);

    struct ggml_cgraph * gf_tiled = ggml_new_graph(ctx);
    struct ggml_tensor * C_tiled  = ggml_mul_mat_tiled(ctx, Bq_std, A);
    ggml_build_forward_expand(gf_tiled, C_tiled);

    struct ggml_cgraph * gf_repack = ggml_new_graph(ctx);
    struct ggml_tensor * C_repack  = ggml_mul_mat(ctx, Bq_rep, A);
    ggml_build_forward_expand(gf_repack, C_repack);

    // Put Bq_rep into the repack buffer before the bulk allocation so the
    // allocator treats it as pre-allocated and leaves it out of the default buffer.
    ggml_backend_buffer_t buf_rep = ggml_backend_buft_alloc_buffer(repack_buft, ggml_nbytes(Bq_rep));
    Bq_rep->buffer = buf_rep;
    Bq_rep->data   = ggml_backend_buffer_get_base(buf_rep);
    ggml_backend_buffer_init_tensor(buf_rep, Bq_rep);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    // Integrity check: repack uses the same quantized weights as std, so the two
    // outputs should agree up to accumulation/activation-quantization noise.
    {
        float * A_data = gen_rand_f32(M * N);
        float * B_data = gen_rand_f32(N * K);
        float * B_T = (float *) malloc(N * K * sizeof(float));
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < K; ++k) {
                B_T[k * N + n] = B_data[n * K + k];
            }
        }
        fill_tensor(A,      A_data, M, N, GGML_TYPE_F32);
        fill_tensor(Bq_std, B_T,    K, N, quant_type);
        // same raw quants; the repack buffer's set_tensor repacks them in-place
        fill_tensor(Bq_rep, B_T,    K, N, quant_type);
        free(A_data); free(B_data); free(B_T);

        ggml_backend_graph_compute(backend, gf_std);
        ggml_backend_graph_compute(backend, gf_repack);

        float * out_std   = (float *) malloc(M * K * sizeof(float));
        float * out_repack = (float *) malloc(M * K * sizeof(float));
        ggml_backend_tensor_get(C_std,    out_std,    0, ggml_nbytes(C_std));
        ggml_backend_tensor_get(C_repack, out_repack, 0, ggml_nbytes(C_repack));

        float max_err, rms_err;
        compare_f32(out_std, out_repack, M * K, &max_err, &rms_err);
        printf("  Integrity (repack vs std): max_err %.6f, rms %.6f\n\n", max_err, rms_err);
        free(out_std); free(out_repack);
    }

    double time_std = 0.0, time_tiled = 0.0, time_repack = 0.0;

    for (int i = 0; i < num_iterations; ++i) {
        // fresh random data every iteration, shared across all three paths
        float * A_data = gen_rand_f32(M * N);
        float * B_data = gen_rand_f32(N * K);
        float * B_T = (float *) malloc(N * K * sizeof(float));
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < K; ++k) {
                B_T[k * N + n] = B_data[n * K + k];
            }
        }
        fill_tensor(A,      A_data, M, N, GGML_TYPE_F32);
        fill_tensor(Bq_std, B_T,    K, N, quant_type);
        fill_tensor(Bq_rep, B_T,    K, N, quant_type);
        free(A_data); free(B_data); free(B_T);

        time_std    += time_graph_compute(backend, gf_std);
        time_tiled  += time_graph_compute(backend, gf_tiled);
        time_repack += time_graph_compute(backend, gf_repack);
    }

    const double flops = 2.0 * M * N * K * num_iterations;
    const double tflops_std    = flops / (time_std    * 1e12);
    const double tflops_tiled  = flops / (time_tiled  * 1e12);
    const double tflops_repack = flops / (time_repack * 1e12);

    printf("BENCH %lldx%lld * %lldx%lld (%s), %d iters:\n",
           (long long)M, (long long)N, (long long)N, (long long)K, ggml_type_name(quant_type), num_iterations);
    printf("  standard:  %8.4f s  %7.3f TFLOPS\n", time_std,    tflops_std);
    printf("  repack:    %8.4f s  %7.3f TFLOPS  (%.2fx vs std)\n", time_repack, tflops_repack, time_std / time_repack);
    printf("  tiled:     %8.4f s  %7.3f TFLOPS  (%.2fx vs std)\n\n", time_tiled, tflops_tiled, time_std / time_tiled);

    ggml_free(ctx);
}

int main(void) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 16);
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_Q6_K);
    test_matmul(backend, 256, 1024, 8192, GGML_TYPE_Q6_K);
    test_matmul(backend, 256, 1024, 8192, GGML_TYPE_Q6_K);
    test_matmul(backend, 357, 1024, 137, GGML_TYPE_Q6_K);
    test_matmul(backend, 16, 1024, 16, GGML_TYPE_Q5_K);
    test_matmul(backend, 18, 1024, 7, GGML_TYPE_Q5_K);
    test_matmul(backend, 18, 1024, 256, GGML_TYPE_Q5_K);
    test_matmul(backend, 256, 1024, 7, GGML_TYPE_Q5_K);
    test_matmul(backend, 16, 1024, 16, GGML_TYPE_Q5_K);

    // probes: N=256 (single QK_K block) vs N=512 (two blocks), per type
    test_matmul(backend, 8, 256, 8, GGML_TYPE_Q5_K);
    test_matmul(backend, 8, 512, 8, GGML_TYPE_Q5_K);
    test_matmul(backend, 8, 256, 8, GGML_TYPE_Q6_K);
    test_matmul(backend, 8, 512, 8, GGML_TYPE_Q6_K);

    // fuzz: small M/K around tile (256) and microtile (16) boundaries
    test_matmul(backend, 1, 1024, 1, GGML_TYPE_Q4_K);
    test_matmul(backend, 2, 1024, 3, GGML_TYPE_Q4_K);
    test_matmul(backend, 15, 1024, 15, GGML_TYPE_Q4_K);
    test_matmul(backend, 17, 1024, 17, GGML_TYPE_Q4_K);
    test_matmul(backend, 31, 1024, 31, GGML_TYPE_Q4_K);
    test_matmul(backend, 33, 1024, 33, GGML_TYPE_Q4_K);
    test_matmul(backend, 47, 1024, 47, GGML_TYPE_Q4_K);
    test_matmul(backend, 255, 1024, 255, GGML_TYPE_Q4_K);
    test_matmul(backend, 257, 1024, 257, GGML_TYPE_Q4_K);
    test_matmul(backend, 271, 1024, 271, GGML_TYPE_Q4_K);
    test_matmul(backend, 272, 1024, 272, GGML_TYPE_Q4_K);
    test_matmul(backend, 511, 1024, 511, GGML_TYPE_Q4_K);
    test_matmul(backend, 513, 1024, 513, GGML_TYPE_Q4_K);
    test_matmul(backend, 33, 1024, 257, GGML_TYPE_Q4_K);
    test_matmul(backend, 257, 1024, 33, GGML_TYPE_Q4_K);
    test_matmul(backend, 17, 512, 17, GGML_TYPE_Q4_K);
    test_matmul(backend, 17, 512, 257, GGML_TYPE_Q4_K);
    test_matmul(backend, 257, 512, 17, GGML_TYPE_Q4_K);

    test_matmul(backend, 1, 1024, 1, GGML_TYPE_Q5_K);
    test_matmul(backend, 2, 1024, 3, GGML_TYPE_Q5_K);
    test_matmul(backend, 15, 1024, 15, GGML_TYPE_Q5_K);
    test_matmul(backend, 17, 1024, 17, GGML_TYPE_Q5_K);
    test_matmul(backend, 31, 1024, 31, GGML_TYPE_Q5_K);
    test_matmul(backend, 33, 1024, 33, GGML_TYPE_Q5_K);
    test_matmul(backend, 47, 1024, 47, GGML_TYPE_Q5_K);
    test_matmul(backend, 255, 1024, 255, GGML_TYPE_Q5_K);
    test_matmul(backend, 257, 1024, 257, GGML_TYPE_Q5_K);
    test_matmul(backend, 271, 1024, 271, GGML_TYPE_Q5_K);
    test_matmul(backend, 272, 1024, 272, GGML_TYPE_Q5_K);
    test_matmul(backend, 511, 1024, 511, GGML_TYPE_Q5_K);
    test_matmul(backend, 513, 1024, 513, GGML_TYPE_Q5_K);
    test_matmul(backend, 33, 1024, 257, GGML_TYPE_Q5_K);
    test_matmul(backend, 257, 1024, 33, GGML_TYPE_Q5_K);
    test_matmul(backend, 17, 512, 17, GGML_TYPE_Q5_K);
    test_matmul(backend, 17, 512, 257, GGML_TYPE_Q5_K);
    test_matmul(backend, 257, 512, 17, GGML_TYPE_Q5_K);

    // bench_matmul_comparison(backend, 8192, GGML_TYPE_Q6_K);
    // bench_matmul_comparison(backend, 8192, GGML_TYPE_Q5_K);
    // bench_matmul_comparison(backend, 8192, GGML_TYPE_Q4_K);

    bench_tiled_vs_std(backend, 2048, 2048, 2048, GGML_TYPE_Q5_K);
    bench_tiled_vs_std(backend, 8192, 1024, 8192, GGML_TYPE_Q6_K);
    bench_tiled_vs_std(backend, 8192, 8192, 8192, GGML_TYPE_Q4_K);

    // three-way: standard vs repacked vs tiled for Q4_K
    bench_three_way(backend, 8192, 8192, 8192, GGML_TYPE_Q4_K);

    // small M (decode-like): weight unpack cost dominates at small M
    bench_tiled_vs_std(backend, 1, 8192, 8192, GGML_TYPE_Q4_K);
    bench_tiled_vs_std(backend, 4, 8192, 8192, GGML_TYPE_Q4_K);
    bench_tiled_vs_std(backend, 16, 8192, 8192, GGML_TYPE_Q4_K);
    bench_tiled_vs_std(backend, 32, 8192, 8192, GGML_TYPE_Q4_K);
    bench_tiled_vs_std(backend, 64, 8192, 8192, GGML_TYPE_Q4_K);

    ggml_backend_free(backend);
    return 0;
}
