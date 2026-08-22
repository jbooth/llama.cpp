
#include "ggml-common.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "tiled.h"
#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"
#endif

#include "ggml-quants.h"
#include "quants.h"
#include "tiled-kernel.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>

#define QK_K 256 // TODO why don't we pick this up properly from headers

static void to_float_f32(const void  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    float * src = (float *) x;
    memcpy(y, x, k * sizeof(float));
}

static void to_float_q8_k(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    dequantize_row_q8_K((const block_q8_K *) x, y, k);
}

static ggml_to_float_t get_to_float_t(ggml_type qtype) {
    const struct ggml_type_traits * trait = ggml_get_type_traits(qtype);
    if (trait->to_float != nullptr) {
        return trait->to_float;
    }
    switch (qtype) {
        case GGML_TYPE_F32:
            return to_float_f32;
        case GGML_TYPE_Q8_K:
            return to_float_q8_k;
    }
    GGML_ASSERT(false);
}

struct TileBuffer {
    float* data = nullptr;

    // Destructor: This is the magic.
    // It runs automatically when the thread terminates.
    ~TileBuffer() {
        if (data) {
            std::free(data);
            // Optional: logging to verify it worked during testing
            // std::cout << "Thread buffer freed!" << std::endl;
        }
    }
};

thread_local TileBuffer src0_buffer;
thread_local TileBuffer src1_buffer;
thread_local TileBuffer dst_buffer;

static void ggml_compute_forward_mul_mat_one_chunk_tiled_explicit_dequant(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {
    //GGML_LOG_INFO("Chunk ir0_start %ld ir0_end %ld ir1_start %ld ir1_end %ld \n", ir0_start, ir0_end, ir1_start, ir1_end);
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
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb13 = src1->nb[3];

    const size_t nb1  = dst->nb[1];
    const size_t nb2  = dst->nb[2];
    const size_t nb3  = dst->nb[3];

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = ggml_get_type_traits_cpu(type)->vec_dot;
    enum ggml_type const vec_dot_type = ggml_get_type_traits_cpu(type)->vec_dot_type;

    const size_t src0_bs = ggml_type_size(type);
    const size_t src1_bs = ggml_type_size(vec_dot_type);

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    GGML_ASSERT(ne00 % QK_K == 0);

    const int64_t TILE = 256;
    const int64_t MICRO = 16;

    float tmp[256]; // partial microtile (used per K block)

    // === logical 256x256 tiles ===
    for (int64_t tile_ir1 = ir1_start; tile_ir1 < ir1_end; tile_ir1 += TILE) {
        for (int64_t tile_ir0 = ir0_start; tile_ir0 < ir0_end; tile_ir0 += TILE) {

            const int64_t tile_ir1_end = MIN(tile_ir1 + TILE, ir1_end);
            const int64_t tile_ir0_end = MIN(tile_ir0 + TILE, ir0_end);
            //GGML_LOG_INFO("Computing tile ir0 %ld ir0_end %ld ir1 %ld ir1_end %d \n", tile_ir0, tile_ir0_end, tile_ir1, tile_ir1_end);
            // Compute tiles of length 256 towards our NXN output block
            for (int64_t ib = 0; ib < ne00; ib += QK_K) {
                for (int64_t iir1 = tile_ir1; iir1 < tile_ir1_end; iir1 += MICRO) {
                    for (int64_t iir0 = tile_ir0; iir0 < tile_ir0_end; iir0 += MICRO) {

                        const int64_t micro_ir1_end = MIN(iir1 + MICRO, tile_ir1_end);
                        const int64_t micro_ir0_end = MIN(iir0 + MICRO, tile_ir0_end);
                        //GGML_LOG_INFO("Computing micro ir0 %ld ir0_end %ld ir1 %ld ir1_end %ld ib %ld \n", iir0, iir1, micro_ir0_end, micro_ir1_end, ib);

                        memset(tmp, 0, sizeof(tmp));

                        for (int64_t ir1 = iir1; ir1 < micro_ir1_end; ++ir1) {

                            const int64_t i13 = ir1 / (ne12 * ne11);
                            const int64_t i12 = (ir1 - i13 * ne12 * ne11) / ne11;
                            const int64_t i11 = ir1 - i13 * ne12 * ne11 - i12 * ne11;

                            const int64_t i03 = i13 / r3;
                            const int64_t i02 = i12 / r2;

                            const char * src0_row =
                                (const char *) src0->data +
                                i02 * nb02 + i03 * nb03 +
                                (ib / QK_K) * src0_bs;

                            const char * src1_col =
                                (const char *) wdata +
                                (src1_cont || src1->type != vec_dot_type
                                    ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                                    : (i11 * nb11 + i12 * nb12 + i13 * nb13));
/*
                            *                    const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03)
                                                  + ib/QK_K * src0_bs;

                                                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                                                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                                                //       the original src1 data pointer, so we should index using the indices directly
                                                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                                                const char * src1_col = (const char*)wdata +
                                                    (src1_cont || src1->type != vec_dot_type
                                                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                                                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                                                src1_col += ib/QK_K * src1_bs;
 */
                            //src1_col += (ib / QK_K) * src1_bs;
                            src1_col += (ib / QK_K) * src1_bs;

                            for (int64_t ir0 = iir0; ir0 < micro_ir0_end; ++ir0) {
                                float partial = 0.0f;
                                //GGML_LOG_INFO("vec_dot row %ld col %ld \n", ir0, i11);
                                vec_dot(
                                    QK_K,
                                    &tmp[(ir0 - iir0) + ((ir1 - iir1) * MICRO)],
                                    0,
                                    src0_row + ir0 * nb01,
                                    0,
                                    src1_col,
                                    0,
                                    1);

                                //tmp[(ir0 - iir0) + ((ir1 - iir1) * MICRO)] = partial;
                            }
                        }

                        // === STAGE 1b: accumulate directly into dst ===
                        //GGML_LOG_INFO("Updating microtile ir0 %ld ir0_end %ld ir1 %ld ir1_end %ld ib %ld \n", iir0, micro_ir0_end, iir1, micro_ir1_end, ib);
                        for (int64_t ir1 = iir1; ir1 < micro_ir1_end; ++ir1) {

                            const int64_t i13 = ir1 / (ne12 * ne11);
                            const int64_t i12 = (ir1 - i13 * ne12 * ne11) / ne11;
                            const int64_t i11 = ir1 - i13 * ne12 * ne11 - i12 * ne11;

                            float * dst_col =
                                (float *) ((char *) dst->data +
                                           i11 * nb1 + i12 * nb2 + i13 * nb3);


                            float * acc = &tmp[(ir1 - iir1) * MICRO];

                            for (int64_t ir0 = iir0; ir0 < micro_ir0_end; ++ir0) {
                                dst_col[ir0] += acc[ir0 - iir0];
                            }
                        }
                    }
                }
            }
        }
    }
}



static void ggml_compute_forward_mul_mat_tiled_explicit_dequant(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int64_t ne00 = (src0) ? (src0)->ne[0] : 0; (void)(ne00);
    const int64_t ne01 = (src0) ? (src0)->ne[1] : 0; (void)(ne01);
    const int64_t ne02 = (src0) ? (src0)->ne[2] : 0; (void)(ne02);
    const int64_t ne03 = (src0) ? (src0)->ne[3] : 0; (void)(ne03);
    const size_t nb00 = (src0) ? (src0)->nb[0] : 0; (void)(nb00);
    const size_t nb01 = (src0) ? (src0)->nb[1] : 0; (void)(nb01);
    const size_t nb02 = (src0) ? (src0)->nb[2] : 0; (void)(nb02);
    const size_t nb03 = (src0) ? (src0)->nb[3] : 0; (void)(nb03);
    const int64_t ne10 = (src1) ? (src1)->ne[0] : 0; (void)(ne10);
    const int64_t ne11 = (src1) ? (src1)->ne[1] : 0; (void)(ne11);
    const int64_t ne12 = (src1) ? (src1)->ne[2] : 0; (void)(ne12);
    const int64_t ne13 = (src1) ? (src1)->ne[3] : 0; (void)(ne13);
    const size_t nb10 = (src1) ? (src1)->nb[0] : 0; (void)(nb10);
    const size_t nb11 = (src1) ? (src1)->nb[1] : 0; (void)(nb11);
    const size_t nb12 = (src1) ? (src1)->nb[2] : 0; (void)(nb12);
    const size_t nb13 = (src1) ? (src1)->nb[3] : 0; (void)(nb13);
    const int64_t ne0 = (dst) ? (dst)->ne[0] : 0; (void)(ne0);
    const int64_t ne1 = (dst) ? (dst)->ne[1] : 0; (void)(ne1);
    const int64_t ne2 = (dst) ? (dst)->ne[2] : 0; (void)(ne2);
    const int64_t ne3 = (dst) ? (dst)->ne[3] : 0; (void)(ne3);
    const size_t nb0 = (dst) ? (dst)->nb[0] : 0; (void)(nb0);
    const size_t nb1 = (dst) ? (dst)->nb[1] : 0; (void)(nb1);
    const size_t nb2 = (dst) ? (dst)->nb[2] : 0; (void)(nb2);
    const size_t nb3 = (dst) ? (dst)->nb[3] : 0; (void)(nb3);

    const int ith = params->ith;
    const int nth = params->nth;
    // if (ith == 1) {
    //     GGML_LOG_INFO("Thread 1 out of %d MATMUL\n", nth);
    //     GGML_LOG_INFO("MATMUL A %s [ %ld , %ld , %ld , %ld ]\n", src0->name, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //     GGML_LOG_INFO("MATMUL B %s [ %ld , %ld , %ld , %ld ]\n", src1->name, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //     GGML_LOG_INFO("MATMUL C %s [ %ld , %ld , %ld , %ld ]\n", dst->name, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    // }
    enum ggml_type           const vec_dot_type         = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
    ggml_from_float_t        const from_float           = ggml_get_type_traits_cpu(vec_dot_type)->from_float;
    int64_t                  const vec_dot_num_rows     = ggml_get_type_traits_cpu(src0->type)->nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    if (src1->type != vec_dot_type) {
        char * wdata = (char *) params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

    #if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    #endif
    }

    if (ith == 0) {
        // result is accumulated with += below, so it must start at zero
        memset(dst->data, 0, nb0 * ne0 * ne1 * ne2 * ne3);
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
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
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // Step down chunk size if too few chunks
    while (nchunk0 * nchunk1 < nth * 4 && chunk_size > 16) {
        chunk_size = chunk_size / 2;
        nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        nchunk1 = (nr1 + chunk_size - 1) / chunk_size;
    }
    //GGML_LOG_INFO("Chunk plan: chunk_size %d chunks %ld \n", chunk_size, nchunk0 * nchunk1);
    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.

    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        //GGML_LOG_INFO("Scrapping chunk plan with chunk_size %d chunks %ld \n", chunk_size, nchunk0 * nchunk1);
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
        //GGML_LOG_INFO("nchunk0 %ld nchunk1 %ld  \n", nchunk0, nchunk1);
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

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk_tiled_explicit_dequant(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}


static void ggml_compute_forward_mul_mat_one_chunk_tiled_implit(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];


    const int64_t ne00 = (src0) ? (src0)->ne[0] : 0; (void)(ne00);
    const int64_t ne01 = (src0) ? (src0)->ne[1] : 0; (void)(ne01);
    const int64_t ne02 = (src0) ? (src0)->ne[2] : 0; (void)(ne02);
    const int64_t ne03 = (src0) ? (src0)->ne[3] : 0; (void)(ne03);
    const size_t nb00 = (src0) ? (src0)->nb[0] : 0; (void)(nb00);
    const size_t nb01 = (src0) ? (src0)->nb[1] : 0; (void)(nb01);
    const size_t nb02 = (src0) ? (src0)->nb[2] : 0; (void)(nb02);
    const size_t nb03 = (src0) ? (src0)->nb[3] : 0; (void)(nb03);
    const int64_t ne10 = (src1) ? (src1)->ne[0] : 0; (void)(ne10);
    const int64_t ne11 = (src1) ? (src1)->ne[1] : 0; (void)(ne11);
    const int64_t ne12 = (src1) ? (src1)->ne[2] : 0; (void)(ne12);
    const int64_t ne13 = (src1) ? (src1)->ne[3] : 0; (void)(ne13);
    const size_t nb10 = (src1) ? (src1)->nb[0] : 0; (void)(nb10);
    const size_t nb11 = (src1) ? (src1)->nb[1] : 0; (void)(nb11);
    const size_t nb12 = (src1) ? (src1)->nb[2] : 0; (void)(nb12);
    const size_t nb13 = (src1) ? (src1)->nb[3] : 0; (void)(nb13);
    const int64_t ne0 = (dst) ? (dst)->ne[0] : 0; (void)(ne0);
    const int64_t ne1 = (dst) ? (dst)->ne[1] : 0; (void)(ne1);
    const int64_t ne2 = (dst) ? (dst)->ne[2] : 0; (void)(ne2);
    const int64_t ne3 = (dst) ? (dst)->ne[3] : 0; (void)(ne3);
    const size_t nb0 = (dst) ? (dst)->nb[0] : 0; (void)(nb0);
    const size_t nb1 = (dst) ? (dst)->nb[1] : 0; (void)(nb1);
    const size_t nb2 = (dst) ? (dst)->nb[2] : 0; (void)(nb2);
    const size_t nb3 = (dst) ? (dst)->nb[3] : 0; (void)(nb3);


    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = ggml_get_type_traits_cpu(type)->vec_dot;
    enum ggml_type const vec_dot_type = ggml_get_type_traits_cpu(type)->vec_dot_type;

    const size_t src0_bs = ggml_type_size(type);
    const size_t src1_bs = ggml_type_size(vec_dot_type);

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //GGML_LOG_INFO("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne00 % QK_K == 0);
    assert(row_size == (ne10 / QK_K) * src1_bs);
    assert(nb01 == ggml_row_size(src0->type, ne00));
    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)

    float tmp[256];
    //memset(tmp, 0, 256 * sizeof(float));

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            memset(tmp, 0, 256 * sizeof(float));
            for (int ib = 0 ; ib < ne00 ; ib += QK_K) {
                for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 ++) {
                    const int64_t i13 = (ir1 / (ne12 * ne1));
                    const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                    const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                    // broadcast src0 into src1
                    const int64_t i03 = i13 / r3;
                    const int64_t i02 = i12 / r2;

                    const int64_t i1 = i11;
                    const int64_t i2 = i12;
                    const int64_t i3 = i13;

                    const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03)
                      + ib/QK_K * src0_bs;

                    // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                    //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                    //       the original src1 data pointer, so we should index using the indices directly
                    // TODO: this is a bit of a hack, we should probably have a better way to handle this
                    const char * src1_col = (const char*)wdata +
                        (src1_cont || src1->type != vec_dot_type
                            ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                            : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                    src1_col += ib/QK_K * src1_bs;

                    float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                    //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                    //}

                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0++) {
                        float partial = 0.0f;
                        vec_dot(QK_K, &partial,
                            0,
                            src0_row + ir0 * nb01,
                            0,
                            src1_col,
                            0,
                            1);
                        tmp[(ir0 - iir0) + ((ir1 - iir1) * 16)] += partial;
                    }
                }
            }
            // Copy out to main memory
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 ++) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));
                float * src_col = &tmp[(ir1 - iir1) * 16];
                memcpy(dst_col + iir0, src_col,(MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
            }
        }
    }
}

void ggml_compute_forward_mul_mat_tiled_implicit(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int64_t ne00 = (src0) ? (src0)->ne[0] : 0; (void)(ne00);
    const int64_t ne01 = (src0) ? (src0)->ne[1] : 0; (void)(ne01);
    const int64_t ne02 = (src0) ? (src0)->ne[2] : 0; (void)(ne02);
    const int64_t ne03 = (src0) ? (src0)->ne[3] : 0; (void)(ne03);
    const size_t nb00 = (src0) ? (src0)->nb[0] : 0; (void)(nb00);
    const size_t nb01 = (src0) ? (src0)->nb[1] : 0; (void)(nb01);
    const size_t nb02 = (src0) ? (src0)->nb[2] : 0; (void)(nb02);
    const size_t nb03 = (src0) ? (src0)->nb[3] : 0; (void)(nb03);
    const int64_t ne10 = (src1) ? (src1)->ne[0] : 0; (void)(ne10);
    const int64_t ne11 = (src1) ? (src1)->ne[1] : 0; (void)(ne11);
    const int64_t ne12 = (src1) ? (src1)->ne[2] : 0; (void)(ne12);
    const int64_t ne13 = (src1) ? (src1)->ne[3] : 0; (void)(ne13);
    const size_t nb10 = (src1) ? (src1)->nb[0] : 0; (void)(nb10);
    const size_t nb11 = (src1) ? (src1)->nb[1] : 0; (void)(nb11);
    const size_t nb12 = (src1) ? (src1)->nb[2] : 0; (void)(nb12);
    const size_t nb13 = (src1) ? (src1)->nb[3] : 0; (void)(nb13);
    const int64_t ne0 = (dst) ? (dst)->ne[0] : 0; (void)(ne0);
    const int64_t ne1 = (dst) ? (dst)->ne[1] : 0; (void)(ne1);
    const int64_t ne2 = (dst) ? (dst)->ne[2] : 0; (void)(ne2);
    const int64_t ne3 = (dst) ? (dst)->ne[3] : 0; (void)(ne3);
    const size_t nb0 = (dst) ? (dst)->nb[0] : 0; (void)(nb0);
    const size_t nb1 = (dst) ? (dst)->nb[1] : 0; (void)(nb1);
    const size_t nb2 = (dst) ? (dst)->nb[2] : 0; (void)(nb2);
    const size_t nb3 = (dst) ? (dst)->nb[3] : 0; (void)(nb3);

    const int ith = params->ith;
    const int nth = params->nth;
    // if (ith == 1) {
    //     GGML_LOG_INFO("Thread 1 out of %d MATMUL\n", nth);
    //     GGML_LOG_INFO("MATMUL A %s [ %ld , %ld , %ld , %ld ]\n", src0->name, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //     GGML_LOG_INFO("MATMUL B %s [ %ld , %ld , %ld , %ld ]\n", src1->name, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //     GGML_LOG_INFO("MATMUL C %s [ %ld , %ld , %ld , %ld ]\n", dst->name, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    // }
    enum ggml_type           const vec_dot_type         = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
    ggml_from_float_t        const from_float           = ggml_get_type_traits_cpu(vec_dot_type)->from_float;
    int64_t                  const vec_dot_num_rows     = ggml_get_type_traits_cpu(src0->type)->nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#if GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = ggml_is_contiguous(src1);

    if (src1_cont) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        return;
    }
UseGgmlGemm1:;
#endif

    if (src1->type != vec_dot_type) {
        char * wdata = (char *) params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

    #if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    #endif
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        ggml_threadpool_chunk_set(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

#if GGML_USE_LLAMAFILE
    if (src1->type != vec_dot_type) {
        const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
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

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk_tiled_implit(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

// === new tiled path (phase 1: q5_K, scalar kernel) ===

struct TiledKernelWs {
    tiled_tile_a_q5_K * a = nullptr;
    tiled_tile_b      * b = nullptr;
    float             * acc = nullptr;

    ~TiledKernelWs() {
        delete a;
        delete b;
        delete[] acc;
    }
};

thread_local TiledKernelWs tiled_ws;

// shape/type gate; anything not supported here runs the old path
static bool ggml_tiled_matmul_supported(const struct ggml_tensor * src0,
                                        const struct ggml_tensor * src1,
                                        const struct ggml_tensor * dst) {
    // phase 1: q5_K weights only
    if (src0->type != GGML_TYPE_Q5_K) {
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
    return true;
}

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
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

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

    const int n_blocks = (int) (ne00 / 256);
    const size_t ldc = nb1 / nb0;

    if (!tiled_ws.a) {
        tiled_ws.a = new tiled_tile_a_q5_K();
    }
    if (!tiled_ws.b) {
        tiled_ws.b = new tiled_tile_b();
    }
    if (!tiled_ws.acc) {
        tiled_ws.acc = new float[TILED_TILE_ROWS * TILED_TILE_ROWS];
    }

    const int64_t a_stride = nb01 / src0_bs;  // blocks between A rows
    const int64_t b_stride = (src1->type == vec_dot_type ? src1->nb[1] : row_size) / src1_bs;

    // slide 256-wide windows across the band, clamped at the band edges and at the
    // ne11 boundaries so the A batch index is constant within a window; advance by the
    // clamped window end (not a fixed 256) so no rows are skipped at a batch boundary
    for (int64_t tile_ir1 = ir1_start; tile_ir1 < ir1_end; ) {
        int64_t tile_ir1_end = MIN(tile_ir1 + 256, ir1_end);
        const int64_t bnd = (tile_ir1 / ne11 + 1) * ne11;
        if (bnd < tile_ir1_end) {
            tile_ir1_end = bnd;
        }

        const int n_b = (int) (tile_ir1_end - tile_ir1);

        // A batch coords (broadcast factors are 1 here)
        const int64_t i13 = tile_ir1 / (ne12 * ne11);
        const int64_t i12 = (tile_ir1 - i13 * ne12 * ne11) / ne11;

        const char * a_base = (const char *) src0->data + i12 * src0->nb[2] + i13 * src0->nb[3];
        char * c_base = (char *) dst->data + i12 * nb2 + i13 * nb3;

        const block_q8_K * b_rows = (const block_q8_K *) ((const char *) wdata + tile_ir1 * b_stride * src1_bs);

        for (int64_t tile_ir0 = ir0_start; tile_ir0 < ir0_end; tile_ir0 += 256) {
            const int64_t tile_ir0_end = MIN(tile_ir0 + 256, ir0_end);
            const int n_a = (int) (tile_ir0_end - tile_ir0);

            const block_q5_K * a_rows = (const block_q5_K *) (a_base + tile_ir0 * nb01);

            float * c = (float *) (c_base + tile_ir0 * nb0 + tile_ir1 * nb1);

            // zero the j-major accumulator once per window, accumulate over all 256-K
            // blocks (each a cheap contiguous buffer add), then transpose-store to C once
            // so every C element is written exactly once
            memset(tiled_ws.acc, 0, (size_t)TILED_TILE_ROWS * TILED_TILE_ROWS * sizeof(float));
            for (int b = 0; b < n_blocks; b++) {
                tiled_unpack_a_q5_K(a_rows + b, a_stride, n_a, tiled_ws.a);
                tiled_unpack_b_q8_K(b_rows + b, b_stride, n_b, tiled_ws.b);
                tiled_run_window_q5_K(*tiled_ws.a, *tiled_ws.b, n_a, n_b, tiled_ws.acc, TILED_TILE_ROWS);
            }
            tiled_store_window(tiled_ws.acc, n_a, n_b, TILED_TILE_ROWS, c, ldc);
        }
        tile_ir1 = tile_ir1_end;
    }
}

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

        ggml_compute_forward_mul_mat_one_chunk_tiled_new(params, dst, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

static void tiled_matmul_gateway(const struct ggml_compute_params * params,
                                 struct ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_Q5_K:
            ggml_compute_forward_mul_mat_tiled_new(params, dst);
            break;
        default:
            break;
    }
}

void ggml_compute_forward_mul_mat_tiled(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {
    if (ggml_tiled_matmul_supported(dst->src[0], dst->src[1], dst)) {
        tiled_matmul_gateway(params, dst);
        return;
    }

    ggml_compute_forward_mul_mat_tiled_explicit_dequant(params, dst);
}
