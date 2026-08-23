
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "tiled.h"

#include "ggml-quants.h"
#include "tiled-kernel.h"

#include <assert.h>
#include <string.h>

#include <new>

#define QK_K 256 // TODO why don't we pick this up properly from headers


// === new tiled path (K-quants, scalar + VNNI kernel) ===

// The B tile and the j-major float accumulator are identical for every A format,
// so they are shared thread_local state; only the A tile differs (SUBBLK changes
// the code density and the per-subblock side tables), so it is held per-format via
// tiled_fmt_tile<Fmt>. A thread processes one op (one format) at a time, so sharing
// acc/b across formats is safe: each window zeroes acc and each chunk rebuilds b.
struct TiledKernelWs {
    tiled_tile_b * b = nullptr;
    float        * acc = nullptr;

    ~TiledKernelWs() {
        delete b;
        // acc was allocated 64B-aligned (std::align_val_t), so free with the
        // matching aligned delete, not delete[]
        if (acc) ::operator delete(acc, std::align_val_t(64));
    }
};

thread_local TiledKernelWs tiled_ws;

template <typename Fmt>
struct TiledA {
    typename tiled_fmt_tile<Fmt>::type * p = nullptr;
    ~TiledA() { delete p; }
};
template <typename Fmt>
static TiledA<Fmt> & tiled_a_holder() {
    static thread_local TiledA<Fmt> h;
    return h;
}

// shape/type gate; anything not supported here runs the old path
static bool ggml_tiled_matmul_supported(const struct ggml_tensor * src0,
                                        const struct ggml_tensor * src1,
                                        const struct ggml_tensor * dst) {
    // K-quant weights only (the gateway switch grows per phase)
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K &&
        src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_Q3_K && src0->type != GGML_TYPE_Q2_K) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_Q8_K) {
        return false;
    }
    // reduction dim must be a multiple of the 256-K tile
    if (src0->ne[0] % 256 != 0 || src1->ne[0] != src0->ne[0]) {
        return false;
    }
    // no permuted src
    if (src0->nb[0] != ggml_type_size(src0->type) || src1->nb[0] != ggml_type_size(src1->type)) {
        return false;
    }
    // no transposed dst
    if (dst->nb[0] != sizeof(float) || dst->nb[0] > dst->nb[1] || dst->nb[1] > dst->nb[2] || dst->nb[2] > dst->nb[3]) {
        return false;
    }
    // no broadcast dims in phase 1
    if (src0->ne[2] == 0 || src0->ne[3] == 0 ||
        src1->ne[2] != src0->ne[2] || src1->ne[3] != src0->ne[3]) {
        return false;
    }
    // B rows must be addressable as contiguous rows (wdata is contiguous by construction)
    if (src1->type == GGML_TYPE_Q8_K && !ggml_is_contiguous(src1)) {
        return false;
    }
    // small M (decode): the weight tile unpack is paid once per op and only amortized
    // over M rows; below this the stock vec_dot/GEMV path wins (no row is processed
    // twice, so there is no reuse to offset the unpack)
    if (src1->ne[1] < 64) {
        return false;
    }
    return true;
}

template <typename Fmt>
static void ggml_compute_forward_mul_mat_one_chunk_tiled_new(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int64_t ne00 = src0->ne[0];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const size_t nb01 = src0->nb[1];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    const enum ggml_type vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
    const size_t src0_bs  = ggml_type_size(src0->type);
    const size_t src1_bs  = ggml_type_size(vec_dot_type);

    GGML_ASSERT(ne00 % 256 == 0);

    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const size_t ldc = nb1 / nb0;

    TiledA<Fmt> & ah = tiled_a_holder<Fmt>();
    if (!ah.p) {
        ah.p = new typename tiled_fmt_tile<Fmt>::type();
    }
    if (!tiled_ws.b) {
        tiled_ws.b = new tiled_tile_b();
    }
    if (!tiled_ws.acc) {
        // 64B aligned: the j-major buffer is read-modify-written with 64B vectors
        // every k-block, and each row is 256 floats (1024B) so every access offset
        // is a multiple of 64B. A 64B-aligned base keeps each 64B RMW inside one
        // cache line (a misaligned 64B store would straddle two lines and dirty both).
        tiled_ws.acc = static_cast<float *>(
            ::operator new(sizeof(float) * (size_t) TILED_TILE_ROWS * TILED_TILE_ROWS,
                           std::align_val_t(64)));
    }

    const int64_t a_stride = nb01 / src0_bs;  // blocks between A rows
    const int64_t b_stride = (src1->type == vec_dot_type ? src1->nb[1] : row_size) / src1_bs;

    const int64_t TILE = 256;
    const int64_t MICRO = 16;

    // === logical 256x256 tiles (macrotiles) ===
    // The ir1 macrotile is additionally clamped at the A batch (ne11) boundary:
    // the tiles require a constant batch index (i12/i13) within one window. Advance
    // by the clamped end (not a fixed 256) so no rows are skipped at a batch boundary.
    for (int64_t tile_ir1 = ir1_start; tile_ir1 < ir1_end; ) {
        int64_t tile_ir1_end = MIN(tile_ir1 + TILE, ir1_end);
        const int64_t bnd = (tile_ir1 / ne11 + 1) * ne11;
        if (bnd < tile_ir1_end) {
            tile_ir1_end = bnd;
        }

        const int n_b = (int) (tile_ir1_end - tile_ir1);

        // A batch coords (constant within the clamped window)
        const int64_t i13 = tile_ir1 / (ne12 * ne11);
        const int64_t i12 = (tile_ir1 - i13 * ne12 * ne11) / ne11;
        // within-batch row; c_base already holds the i12/i13 batch offset, so the
        // row offset must use i11 (not the flattened tile_ir1, which spans all batch dims)
        const int64_t i11 = tile_ir1 - i13 * ne12 * ne11 - i12 * ne11;

        // dst batches == src1 batches (ggml_mul_mat_tiled), so the loop batch
        // coords are in src1 space; src0 batches are broadcast over them
        // (ne02|ne12, ne03|ne13), map down into src0's own batch for the A tile
        const int64_t r2 = ne12 / ne02;
        const int64_t r3 = ne13 / ne03;
        const int64_t i02 = i12 / r2;
        const int64_t i03 = i13 / r3;

        const char * a_base = (const char *) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3];
        char * c_base = (char *) dst->data + i12 * nb2 + i13 * nb3;

        const block_q8_K * b_rows = (const block_q8_K *) ((const char *) wdata + tile_ir1 * b_stride * src1_bs);

        for (int64_t tile_ir0 = ir0_start; tile_ir0 < ir0_end; tile_ir0 += TILE) {
            const int64_t tile_ir0_end = MIN(tile_ir0 + TILE, ir0_end);
            const int n_a = (int) (tile_ir0_end - tile_ir0);

            const char * a_rows = a_base + tile_ir0 * nb01;

            float * c = (float *) (c_base + tile_ir0 * nb0 + i11 * nb1);

            // j-major buffer: zeroed once per macrotile, accumulated over all slabs and
            // microtiles (each a cheap contiguous add), then transposed into C once so
            // every C element is written exactly once (a per-microtile strided C RMW
            // measured ~30% of runtime)
            memset(tiled_ws.acc, 0, (size_t)TILED_TILE_ROWS * TILED_TILE_ROWS * sizeof(float));

            // Compute tiles of length 256 towards our NXN output block
            for (int64_t ib = 0; ib < ne00; ib += TILE) {
                const int b = (int) (ib / TILE);
                tiled_unpack_a(Fmt(), a_rows + b * src0_bs, a_stride, n_a, ah.p);
                tiled_unpack_b_q8_K(b_rows + b, b_stride, n_b, tiled_ws.b);

                for (int64_t iir1 = tile_ir1; iir1 < tile_ir1_end; iir1 += MICRO) {
                    for (int64_t iir0 = tile_ir0; iir0 < tile_ir0_end; iir0 += MICRO) {
                        // 16x16 microtile window over the macrotile (tile-local coords);
                        // the kernel processes the full 16x16 unconditionally, rows/cols
                        // past the window edges hold harmless tile garbage (B rows are
                        // zero-padded at unpack) and the store below drops them
                        tiled_run_microtile(*ah.p, *tiled_ws.b,
                            (int) (iir0 - tile_ir0), (int) (iir1 - tile_ir1),
                            tiled_ws.acc, TILED_TILE_ROWS);
                    }
                }
            }

            tiled_store_window(tiled_ws.acc, n_a, n_b, TILED_TILE_ROWS, c, ldc);
        }
        tile_ir1 = tile_ir1_end;
    }
}

template <typename Fmt>
static void ggml_compute_forward_mul_mat_tiled_new(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const size_t nb10  = src1->nb[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];
    const size_t nb0  = dst->nb[0];
    const size_t nb1  = dst->nb[1];
    const size_t nb2  = dst->nb[2];
    const size_t nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type      const vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
    ggml_from_float_t   const from_float   = ggml_get_type_traits_cpu(vec_dot_type)->from_float;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    if (src1->type != vec_dot_type) {
        char * wdata = (char *) params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*src1->nb[3] + i12*src1->nb[2] + i11*src1->nb[1] + ne10_block_start*bs*src1->nb[0]),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    }

    if (ith == 0) {
        // result is accumulated with += below, so it must start at zero
        memset(dst->data, 0, nb0 * ne0 * ne1 * ne2 * ne3);
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth. This saves a bit of coordination right at the start.
        ggml_threadpool_chunk_set(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 256;

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim. CEIL(nr/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // Step down chunk size if too few chunks
    while (nchunk0 * nchunk1 < nth * 4 && chunk_size > 16) {
        chunk_size = chunk_size / 2;
        nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        nchunk1 = (nr1 + chunk_size - 1) / chunk_size;
    }

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        ggml_compute_forward_mul_mat_one_chunk_tiled_new<Fmt>(params, dst, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

static void tiled_matmul_gateway(const struct ggml_compute_params * params,
                                 struct ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_Q4_K:
            ggml_compute_forward_mul_mat_tiled_new<tiled_fmt_q4_K>(params, dst);
            break;
        case GGML_TYPE_Q5_K:
            ggml_compute_forward_mul_mat_tiled_new<tiled_fmt_q5_K>(params, dst);
            break;
        case GGML_TYPE_Q6_K:
            ggml_compute_forward_mul_mat_tiled_new<tiled_fmt_q6_K>(params, dst);
            break;
        case GGML_TYPE_Q3_K:
            ggml_compute_forward_mul_mat_tiled_new<tiled_fmt_q3_K>(params, dst);
            break;
        case GGML_TYPE_Q2_K:
            ggml_compute_forward_mul_mat_tiled_new<tiled_fmt_q2_K>(params, dst);
            break;
        default:
            break;
    }
}

void ggml_compute_forward_mul_mat_tiled(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    tiled_matmul_gateway(params, dst);
}
