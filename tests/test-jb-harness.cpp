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

static double time_graph_compute(ggml_backend_t backend, struct ggml_cgraph * gf) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ggml_backend_graph_compute(backend, gf);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
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

struct bench_row {
    const char * name;
    double time_std, time_repack, time_tiled;
    float max_err_repack, rmse_repack;
    float max_err_tiled, rmse_tiled;
    bool have_repack;
};

// One timed run of each path (standard ggml_mul_mat, repacked ggml_mul_mat
// via the CPU_REPACK buffer, ggml_mul_mat_tiled) on the same quantized
// weights, plus max error and RMSE vs the standard output. std and tiled
// share one weight tensor; repack uses a second tensor in the repack buffer
// (whose set_tensor repacks the raw quants, and whose kernel is selected
// because the weight lives in that buffer type). The repack column is only
// available where a repack kernel exists for the type (x86: q4_K, q2_K);
// init_tensor sets Bq_rep->extra to the repack layout or NULL.
static bench_row bench_three_way(ggml_backend_t backend, int64_t M, int64_t N, int64_t K, ggml_type quant_type) {
    bench_row row;
    row.name = ggml_type_name(quant_type);
    row.time_std = row.time_repack = row.time_tiled = 0.0;
    row.max_err_repack = row.rmse_repack = row.max_err_tiled = row.rmse_tiled = 0.0f;
    row.have_repack = false;

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
    // repack 8x8 layout requires N (ne0) and K (ne1) to be multiples of 8.
    ggml_backend_buffer_type_t repack_buft = get_cpu_repack_buft();
    if (repack_buft && N % 8 == 0 && K % 8 == 0) {
        ggml_backend_buffer_t buf_rep = ggml_backend_buft_alloc_buffer(repack_buft, ggml_nbytes(Bq_rep));
        Bq_rep->buffer = buf_rep;
        Bq_rep->data   = ggml_backend_buffer_get_base(buf_rep);
        ggml_backend_buffer_init_tensor(buf_rep, Bq_rep);
        row.have_repack = (Bq_rep->extra != NULL);
    }

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    float * A_data = gen_rand_f32(M * N);
    float * B_data = gen_rand_f32(N * K);
    // Bq is K rows of N in ggml layout, so transpose B into it
    float * B_T = (float *) malloc(N * K * sizeof(float));
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            B_T[k * N + n] = B_data[n * K + k];
        }
    }
    fill_tensor(A,      A_data, M, N, GGML_TYPE_F32);
    fill_tensor(Bq_std, B_T,    K, N, quant_type);
    if (row.have_repack) {
        // same raw quants; the repack buffer's set_tensor repacks them in-place
        fill_tensor(Bq_rep, B_T,    K, N, quant_type);
    }
    free(A_data); free(B_data); free(B_T);

    // one timing per path
    row.time_std    = time_graph_compute(backend, gf_std);
    if (row.have_repack) {
        row.time_repack = time_graph_compute(backend, gf_repack);
    }
    row.time_tiled  = time_graph_compute(backend, gf_tiled);

    // errors vs the standard output (all paths use the same quantized weights)
    float * out_std    = (float *) malloc(M * K * sizeof(float));
    float * out_tiled  = (float *) malloc(M * K * sizeof(float));
    float * out_repack = (float *) malloc(M * K * sizeof(float));
    ggml_backend_tensor_get(C_std,   out_std,   0, ggml_nbytes(C_std));
    ggml_backend_tensor_get(C_tiled, out_tiled, 0, ggml_nbytes(C_tiled));
    if (row.have_repack) {
        ggml_backend_tensor_get(C_repack, out_repack, 0, ggml_nbytes(C_repack));
        compare_f32(out_std, out_repack, M * K, &row.max_err_repack, &row.rmse_repack);
    }
    compare_f32(out_std, out_tiled, M * K, &row.max_err_tiled, &row.rmse_tiled);
    free(out_std); free(out_tiled); free(out_repack);

    ggml_free(ctx);
    return row;
}

static void print_bench_table(int64_t M, int64_t N, int64_t K, const bench_row * rows, size_t n_types) {
    const double flops = 2.0 * M * N * K;

    printf("\nBENCH %lldx%lld * %lldx%lld, one timing per path, 16 threads\n",
           (long long)M, (long long)N, (long long)N, (long long)K);
    printf("%-8s %10s %12s %12s %11s %11s %15s %15s %15s %15s\n",
           "type", "std TF", "repack TF", "tiled TF", "repack/std", "tiled/std",
           "max_err(repack)", "rmse(repack)", "max_err(tiled)", "rmse(tiled)");
    for (size_t i = 0; i < n_types; ++i) {
        const bench_row * r = &rows[i];
        if (r->have_repack) {
            printf("%-8s %10.3f %12.3f %12.3f %11.2fx %11.2fx %15.6e %15.6e %15.6e %15.6e\n",
                   r->name,
                   flops / (r->time_std * 1e12),
                   flops / (r->time_repack * 1e12), flops / (r->time_tiled * 1e12),
                   r->time_std / r->time_repack, r->time_std / r->time_tiled,
                   r->max_err_repack, r->rmse_repack, r->max_err_tiled, r->rmse_tiled);
        } else {
            printf("%-8s %10.3f %12s %12.3f %11s %11.2fx %15s %15s %15.6e %15.6e\n",
                   r->name,
                   flops / (r->time_std * 1e12),
                   "n/a", flops / (r->time_tiled * 1e12), "n/a",
                   r->time_std / r->time_tiled,
                   "n/a", "n/a", r->max_err_tiled, r->rmse_tiled);
        }
    }
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

    // q3_K / q2_K (SUBBLK=16): validate the new unpackers + BIAS/min correction.
    // M>=64 engages the tiled path; the odd K probes the ragged K-tile edge.
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_Q3_K);
    test_matmul(backend, 256, 1024, 8192, GGML_TYPE_Q3_K);
    test_matmul(backend, 357, 1024, 137, GGML_TYPE_Q3_K);
    test_matmul(backend, 8, 256, 8, GGML_TYPE_Q3_K);
    test_matmul(backend, 8, 512, 8, GGML_TYPE_Q3_K);
    test_matmul(backend, 17, 1024, 257, GGML_TYPE_Q3_K);
    test_matmul(backend, 257, 1024, 17, GGML_TYPE_Q3_K);
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_Q2_K);
    test_matmul(backend, 256, 1024, 8192, GGML_TYPE_Q2_K);
    test_matmul(backend, 357, 1024, 137, GGML_TYPE_Q2_K);
    test_matmul(backend, 8, 256, 8, GGML_TYPE_Q2_K);
    test_matmul(backend, 8, 512, 8, GGML_TYPE_Q2_K);
    test_matmul(backend, 17, 1024, 257, GGML_TYPE_Q2_K);
    test_matmul(backend, 257, 1024, 17, GGML_TYPE_Q2_K);

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

    // one timing per quant type and shape; the table compares standard,
    // repack and tiled, with max error / RMSE vs the standard output
    const ggml_type bench_types[] = { GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K };
    const size_t n_types = sizeof(bench_types) / sizeof(bench_types[0]);
    struct { int64_t M, N, K; } shapes[] = {
        { 4096, 4096, 4096 },
        { 4096, 4096,   64 },
    };
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
        bench_row rows[n_types];
        for (size_t i = 0; i < n_types; ++i) {
            rows[i] = bench_three_way(backend, shapes[s].M, shapes[s].N, shapes[s].K, bench_types[i]);
        }
        print_bench_table(shapes[s].M, shapes[s].N, shapes[s].K, rows, n_types);
    }

    ggml_backend_free(backend);
    return 0;
}
