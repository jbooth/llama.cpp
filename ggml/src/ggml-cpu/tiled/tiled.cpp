
#include "ggml-common.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "tiled.h"
#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"
#endif

#include <assert.h>
#include <atomic>
#include <stdlib.h>
#include <string.h>

#define QK_K 256 // TODO why don't we pick this up properly from headers

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