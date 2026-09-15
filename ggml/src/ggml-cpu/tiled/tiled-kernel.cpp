// mulmat microtile kernels

#include "tiled-kernel.h"

#include "ggml.h"

#include <string.h>

#if defined(__AVX512VNNI__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif


// Reference implementation, slower than existing vec_dot approach
template <int SUBBLK, bool HAS_MIN, int BIAS>
static void tiled_run_microtile_scalar(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                       int i0, int j0, int n_cols, float * buf, int buf_stride) {
    constexpr int NB = TILED_TILE_K / SUBBLK;    // subblocks per 256-K block
    constexpr int NS = SUBBLK / 16; // per-16 bsums per subblock

    float acc[TILED_MICRO][TILED_MICRO];
    memset(acc, 0, sizeof(acc));

    // subdots at subblock granularity over the 256-K block, exact integer math
    for (int s = 0; s < NB; s++) {
        for (int j = 0; j < n_cols; j++) {
            const int br = j0 + j;
            const uint8_t * q1 = &src1.q[br * TILED_TILE_K + s * SUBBLK];
            int32_t bsum_s1 = 0;
            for (int u = 0; u < NS; u++) {
                bsum_s1 += src1.bsums[(s * NS + u) * TILED_TILE_ROWS + br];
            }

            for (int i = 0; i < TILED_MICRO; i++) {
                const int ar = i0 + i;
                const int8_t * q0 = (const int8_t *) &src0.q[ar * TILED_TILE_K + s * SUBBLK];

                // raw = sum((s1+128) * true_src0)
                int32_t raw = 0;
                for (int e = 0; e < SUBBLK; e++) {
                    raw += (int32_t) q1[e] * (int32_t) q0[e];
                }

                // correction: raw - 128 * bsums_s0
                int32_t bs0 = 0;
                for (int u = 0; u < NS; u++) { bs0 += src0.bsums[(s * NS + u) * TILED_TILE_ROWS + ar]; }
                int32_t corr = raw - 128 * bs0;

                const int32_t scales_raw = (int32_t) src0.scales[ar * NB + s] * corr;
                if constexpr (HAS_MIN) {
                    const int32_t mins_bsum = (int32_t) src0.mins[ar * NB + s] * bsum_s1;
                    acc[i][j] += (float) src0.d[ar] * (float) scales_raw
                              - (float) src0.dmin[ar] * (float) mins_bsum;
                } else {
                    acc[i][j] += (float) src0.d[ar] * (float) scales_raw;
                }
            }
        }
    }

    // Apply d and write out to buf
    for (int i = 0; i < TILED_MICRO; i++) {
        for (int j = 0; j < n_cols; j++) {
            buf[(j0 + j) * buf_stride + (i0 + i)] += src1.d[j0 + j] * acc[i][j];
        }
    }
}

#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)


// VNNI microkernel
// 8x16 band pass: 8 src0 rows (i0, i0+8) x 16 src1 cols (j0, j0+16)
// Math: s1_acc += scales_s*raw_s, s2_acc += mins_s*bsums_s per subblock; the row result
// is d0*s1_acc - dmin*s2_acc (f32, exact in range), applied once per row.
//
// Register pressure: the band pass holds acc16 + s1_acc = 16 zmm for the
// 16 src0 rows x 8 src1 cols. The 512-bit load covers all 16 src0 rows (interleaved);
// the broadcast is one src1 col at a time. acc16[t] has 16 lanes (one per src0 row).
template <int SUBBLK, bool HAS_MIN, int BIAS, bool FULL, int COLS = 8>
static void tiled_run_micro_vnni_16x8(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                       int i0, int j0, int n_cols, float * buf, int buf_stride,
                                       int k_extent, int slab_offset, int table_stride, int i0_buf) {
    constexpr int NB = TILED_TILE_K / SUBBLK;
    constexpr int NS = SUBBLK / 16;
    constexpr int NG = SUBBLK / 4;
    constexpr int NUM_COLS = COLS; // src1 cols per pass

    if (n_cols == 0) return;

    // FULL (the common GEMM case, n_cols == 8) uses the compile-time constant bound so the
    // dpbusd t-chain stays fully unrolled; the ragged/GEMV path (n_cols < 8) uses the runtime
    // bound to skip work on the zero-padded cols
    const int tmax = FULL ? NUM_COLS : n_cols;

    // extended-K geometry: k_extent is the q row stride (TILED_TILE_K for the normal 256-K
    // slab), slab_offset selects the 256-K slab within the full-K tile, table_stride is the
    // transposed side-table row stride, i0_buf is the acc row (i0 for the normal path)
    const int s16_base = slab_offset * (TILED_TILE_K / 16); // per-16-k group base across the full K
    const int d_base   = slab_offset * TILED_MICRO;         // d/dmin row base (slab * 16 rows)

    __m512i s1_acc[NUM_COLS];
    for (int t = 0; t < tmax; t++) { s1_acc[t] = _mm512_setzero_si512(); }
    __m512i s2_acc[NUM_COLS];
    if constexpr (HAS_MIN) {
        for (int t = 0; t < tmax; t++) { s2_acc[t] = _mm512_setzero_si512(); }
    }

    // src0 bsums for the 16 rows at each subblock: [s * ROWS + row]
    // src1 bsums for the 8 cols: [s * NS * ROWS + col], combined over NS

    for (int s = 0; s < NB; s++) {
        __m512i acc16[NUM_COLS];
        for (int t = 0; t < tmax; t++) { acc16[t] = _mm512_setzero_si512(); }
        // 512-bit load of src0 bsums for 16 rows, combined over NS groups per subblock
        __m512i src0_bsums_16 = _mm512_load_si512((const __m512i *) &src0.bsums[(s16_base + s * NS) * table_stride + i0]);
        for (int u = 1; u < NS; u++) {
            src0_bsums_16 = _mm512_add_epi32(src0_bsums_16, _mm512_load_si512((const __m512i *) &src0.bsums[(s16_base + s * NS + u) * table_stride + i0]));
        }
        __m512i bias_16 = _mm512_mullo_epi32(src0_bsums_16, _mm512_set1_epi32(128));

        for (int g = 0; g < NG; g++) {
            const int kg = s * NG + g;
            // 512-bit load: 16 src0 rows x 4 k, interleaved layout
            // in-place transpose layout: group-local, row = kg%16, chunk c = kg/16; the chunk
            // index is offset by the slab (c_full = slab * 4 + kg/16) and the row stride is k_extent
            const int grp_base = (i0 / TILED_MICRO) * (TILED_MICRO * k_extent);
            const int c_full   = slab_offset * (TILED_TILE_K / (TILED_MICRO * 4)) + (kg / TILED_MICRO);
            const __m512i src0_512 = _mm512_load_si512((const __m512i *) &src0.q[grp_base + (kg % TILED_MICRO) * k_extent + c_full * (TILED_MICRO * 4)]);
            // broadcast: one src1 col's 4 k-values (biased s1+128, uint8)
            for (int t = 0; t < tmax; t++) {
                const uint32_t s1_u4 = *(const uint32_t *) &src1.q[(j0 + t) * k_extent + slab_offset * TILED_TILE_K + kg * 4];
                const __m512i s1_bcast = _mm512_set1_epi32((int) s1_u4);
                acc16[t] = _mm512_dpbusd_epi32(acc16[t], s1_bcast, src0_512);
            }
        }

        // int correction: raw' = dot(s1+128, src0) = dot(s1, src0) + 128*src0_bsums
        // s1_acc += scales * (raw' - 128*src0_bsums)
        const int s_full = slab_offset * NB + s; // subblock index across the full K
        const __m512i scales_16 = _mm512_load_si512((const __m512i *) &src0.scales_t[s_full * table_stride + i0]);
        for (int t = 0; t < tmax; t++) {
            s1_acc[t] = _mm512_add_epi32(s1_acc[t], _mm512_mullo_epi32(_mm512_sub_epi32(acc16[t], bias_16), scales_16));
        }
    }

    // s2_acc += mins * src1_bsums (HAS_MIN only)
    if constexpr (HAS_MIN) {
        for (int s = 0; s < NB; s++) {
            int32_t s1_bsums[NUM_COLS];
            for (int t = 0; t < tmax; t++) {
                int32_t sum = 0;
                for (int u = 0; u < NS; u++) { sum += src1.bsums[(s16_base + s * NS + u) * table_stride + j0 + t]; }
                s1_bsums[t] = sum;
            }
            const int s_full = slab_offset * NB + s;
            const __m512i mins_16 = _mm512_load_si512((const __m512i *) &src0.mins_t[s_full * table_stride + i0]);
            for (int t = 0; t < tmax; t++) {
                s2_acc[t] = _mm512_add_epi32(s2_acc[t], _mm512_mullo_epi32(mins_16, _mm512_set1_epi32(s1_bsums[t])));
            }
        }
    }

    // epilogue: buf is [col][row], so each col's 16 rows are contiguous (64B store)
    const __m512 d0_vec = _mm512_load_ps(&src0.d[d_base + i0]);
    if constexpr (HAS_MIN) {
        const __m512 dmin_vec = _mm512_load_ps(&src0.dmin[d_base + i0]);
        for (int t = 0; t < tmax; t++) {
            __m512 f1 = _mm512_cvtepi32_ps(s1_acc[t]);
            __m512 result = _mm512_mul_ps(f1, d0_vec);
            __m512 f2 = _mm512_cvtepi32_ps(s2_acc[t]);
            result = _mm512_fnmadd_ps(dmin_vec, f2, result);
            float * p = &buf[(j0 + t) * buf_stride + i0_buf];
            _mm512_store_ps(p, _mm512_add_ps(_mm512_load_ps(p), _mm512_mul_ps(result, _mm512_set1_ps(src1.d[d_base + j0 + t]))));
        }
    } else {
        for (int t = 0; t < tmax; t++) {
            __m512 f1 = _mm512_cvtepi32_ps(s1_acc[t]);
            __m512 result = _mm512_mul_ps(f1, d0_vec);
            float * p = &buf[(j0 + t) * buf_stride + i0_buf];
            _mm512_store_ps(p, _mm512_add_ps(_mm512_load_ps(p), _mm512_mul_ps(result, _mm512_set1_ps(src1.d[d_base + j0 + t]))));
        }
    }
}

// 16x16 microtile split into 16xCOLS passes over src1 cols j0..j0+15
template <int SUBBLK, bool HAS_MIN, int BIAS>
static void tiled_run_microtile_vnni(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                         int i0, int j0, int n_cols, float * buf, int buf_stride,
                                         int k_extent, int slab_offset, int table_stride, int i0_buf) {
    // SUBBLK=32 uses 4-col passes: at 8 cols GCC emits a home/active register ping-pong
    // for the dpbusd accumulators (35-39% vec-copy); 4 cols keeps them in dedicated regs.
    // SUBBLK=16 stays at 8 cols (already clean; 4 cols would 4x the per-kg src0 loads).
    constexpr int COLS = (SUBBLK == 32) ? 4 : 8;
    for (int c = 0; c < n_cols; c += COLS) {
        const int nc = COLS < (n_cols - c) ? COLS : (n_cols - c);
        if (nc == COLS) tiled_run_micro_vnni_16x8<SUBBLK, HAS_MIN, BIAS, true , COLS>(src0, src1, i0, j0 + c, nc, buf, buf_stride, k_extent, slab_offset, table_stride, i0_buf);
        else            tiled_run_micro_vnni_16x8<SUBBLK, HAS_MIN, BIAS, false, COLS>(src0, src1, i0, j0 + c, nc, buf, buf_stride, k_extent, slab_offset, table_stride, i0_buf);
    }
}

// GEMV: 1 activation col x 16 weight rows (Driver C). Same extended-tile geometry and
// addressing as the ext MAC (k_extent, slab_offset, table_stride, i0_buf), one call covers
// one 256-K slab. The ILP unit is the k-group WITHIN a subblock (NG independent dpbusd,
// one per 4-k group), not the col: the 16x16 MAC's ILP is the col (acc16[t]), so at n_cols=1
// its g-loop is a serial dpbusd chain. Here acc[g] is per k-group, so the g-loop is NG
// independent dpbusd, then a reduction to the full subblock dot product (one per src0 row).
// The per-subblock scale correction is applied after the reduction (it cannot be factored
// out across k-groups because the scale is shared, so the raw int sum must complete first).
// Bit-identical to the per-slab path: the per-subblock, per-slab accumulation order matches;
// the int reduction is exact.
template <int SUBBLK, bool HAS_MIN, int BIAS>
static void tiled_run_gemv_vnni(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                int j0, float * buf, int buf_stride,
                                int k_extent, int slab_offset, int table_stride, int i0_buf) {
    constexpr int NB = TILED_TILE_K / SUBBLK;   // subblocks per 256-K slab
    constexpr int NS = SUBBLK / 16;            // per-16 bsums per subblock
    constexpr int NG = SUBBLK / 4;             // k-groups (4-k) per subblock

    // ext tile geometry (same as the ext MAC): the 16 weight rows sit at tile rows 0..15
    // (i0 = 0); i0_buf is the acc row offset (the driver's window row offset)
    const int s16_base = slab_offset * (TILED_TILE_K / 16); // per-16-k group base across the chunk
    const int d_base   = slab_offset * TILED_MICRO;         // d/dmin row base (slab * 16 rows)

    __m512i s1_acc = _mm512_setzero_si512();
    __m512i s2_acc = _mm512_setzero_si512();

    for (int s = 0; s < NB; s++) {
        // NG k-group accumulators (the ILP): one dpbusd each, independent across g
        __m512i acc[NG];
        for (int g = 0; g < NG; g++) { acc[g] = _mm512_setzero_si512(); }

        // src0 bsums for the 16 rows at each subblock, combined over the NS per-16 groups
        __m512i src0_bsums_16 = _mm512_load_si512((const __m512i *) &src0.bsums[(s16_base + s * NS) * table_stride + 0]);
        for (int u = 1; u < NS; u++) {
            src0_bsums_16 = _mm512_add_epi32(src0_bsums_16, _mm512_load_si512((const __m512i *) &src0.bsums[(s16_base + s * NS + u) * table_stride + 0]));
        }
        __m512i bias_16 = _mm512_mullo_epi32(src0_bsums_16, _mm512_set1_epi32(128));

        for (int g = 0; g < NG; g++) {
            const int kg = s * NG + g;
            // ext src0 code layout: [kg%16 row][c_full chunk], row stride k_extent (i0 = 0)
            const int c_full   = slab_offset * (TILED_TILE_K / (TILED_MICRO * 4)) + (kg / TILED_MICRO);
            const __m512i src0_512 = _mm512_load_si512((const __m512i *) &src0.q[(kg % TILED_MICRO) * k_extent + c_full * (TILED_MICRO * 4)]);
            // one col's 4 k-values (biased s1+128, uint8); broadcast across the 16 rows
            const uint32_t s1_u4 = *(const uint32_t *) &src1.q[j0 * k_extent + slab_offset * TILED_TILE_K + kg * 4];
            const __m512i s1_bcast = _mm512_set1_epi32((int) s1_u4);
            acc[g] = _mm512_dpbusd_epi32(acc[g], s1_bcast, src0_512);
        }

        // reduce the NG k-groups to the full subblock dot product (one per src0 row)
        __m512i acc0 = acc[0];
        for (int g = 1; g < NG; g++) { acc0 = _mm512_add_epi32(acc0, acc[g]); }

        // int correction: raw' = dot(s1+128, src0); s1_acc += scales * (raw' - 128*src0_bsums)
        const int s_full = slab_offset * NB + s; // subblock index across the chunk
        const __m512i scales_16 = _mm512_load_si512((const __m512i *) &src0.scales_t[s_full * table_stride + 0]);
        s1_acc = _mm512_add_epi32(s1_acc, _mm512_mullo_epi32(_mm512_sub_epi32(acc0, bias_16), scales_16));

        if constexpr (HAS_MIN) {
            // src1 bsums for the single col (a scalar), combined over the NS per-16 groups
            int32_t sum = 0;
            for (int u = 0; u < NS; u++) { sum += src1.bsums[(s16_base + s * NS + u) * table_stride + j0]; }
            const __m512i mins_16 = _mm512_load_si512((const __m512i *) &src0.mins_t[s_full * table_stride + 0]);
            s2_acc = _mm512_add_epi32(s2_acc, _mm512_mullo_epi32(mins_16, _mm512_set1_epi32(sum)));
        }
    }

    // epilogue: 16 outputs (one per src0 row); buf is [col][row], the 16 rows are contiguous
    const __m512 d0_vec = _mm512_load_ps(&src0.d[d_base + 0]);
    __m512 f1 = _mm512_cvtepi32_ps(s1_acc);
    __m512 result = _mm512_mul_ps(f1, d0_vec);
    if constexpr (HAS_MIN) {
        const __m512 dmin_vec = _mm512_load_ps(&src0.dmin[d_base + 0]);
        __m512 f2 = _mm512_cvtepi32_ps(s2_acc);
        result = _mm512_fnmadd_ps(dmin_vec, f2, result);
    }
    float * p = &buf[j0 * buf_stride + i0_buf];
    _mm512_store_ps(p, _mm512_add_ps(_mm512_load_ps(p), _mm512_mul_ps(result, _mm512_set1_ps(src1.d[d_base + j0]))));
}

// L1-resident GEMV (Driver C v2): one 256-block of src0 (16 rows x 256 K) against the
// L1-resident src1 (1 row x ne00). Same addressing as the ext GEMV, but src0 and src1
// have different row strides (k_extent_s0 for the 1024-K window, k_extent_s1 for the
// full ne00). The epilogue accumulates into acc_f (a __m512 of 16 float outputs) instead
// of storing to buf; the caller persists acc_f across K-windows and writes it once per
// 16-row weight window. Per-256-block epilogue (d0*d1 applied per block), so the result
// is bit-identical to the per-slab GEMV.
template <int SUBBLK, bool HAS_MIN, int BIAS>
static void tiled_run_gemv_l1_vnni(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                   int slab_local, int slab_s1, __m512 * acc_f,
                                   int k_extent_s0, int k_extent_s1) {
    constexpr int NB = TILED_TILE_K / SUBBLK;
    constexpr int NS = SUBBLK / 16;
    constexpr int NG = SUBBLK / 4;

    const int s16_s0 = slab_local * (TILED_TILE_K / 16); // src0 per-16 base (within the window)
    const int s16_s1 = slab_s1 * (TILED_TILE_K / 16);    // src1 per-16 base (absolute)
    const int d0_base = slab_local * TILED_MICRO;        // src0 d row base
    const int d1_base = slab_s1 * TILED_MICRO;           // src1 d row base

    __m512i s1_acc = _mm512_setzero_si512();
    __m512i s2_acc = _mm512_setzero_si512();

    for (int s = 0; s < NB; s++) {
        __m512i acc[NG];
        for (int g = 0; g < NG; g++) { acc[g] = _mm512_setzero_si512(); }

        // src0 bsums (slab_local, 16-row table stride)
        __m512i src0_bsums_16 = _mm512_load_si512((const __m512i *) &src0.bsums[(s16_s0 + s * NS) * TILED_MICRO + 0]);
        for (int u = 1; u < NS; u++) {
            src0_bsums_16 = _mm512_add_epi32(src0_bsums_16, _mm512_load_si512((const __m512i *) &src0.bsums[(s16_s0 + s * NS + u) * TILED_MICRO + 0]));
        }
        __m512i bias_16 = _mm512_mullo_epi32(src0_bsums_16, _mm512_set1_epi32(128));

        for (int g = 0; g < NG; g++) {
            const int kg = s * NG + g;
            // src0: [kg%16 row][c_full chunk], row stride k_extent_s0 (i0 = 0)
            const int c_full = slab_local * (TILED_TILE_K / (TILED_MICRO * 4)) + (kg / TILED_MICRO);
            const __m512i src0_512 = _mm512_load_si512((const __m512i *) &src0.q[(kg % TILED_MICRO) * k_extent_s0 + c_full * (TILED_MICRO * 4)]);
            // src1: 1 col (j0=0), row stride k_extent_s1, absolute slab
            const uint32_t s1_u4 = *(const uint32_t *) &src1.q[0 * k_extent_s1 + slab_s1 * TILED_TILE_K + kg * 4];
            const __m512i s1_bcast = _mm512_set1_epi32((int) s1_u4);
            acc[g] = _mm512_dpbusd_epi32(acc[g], s1_bcast, src0_512);
        }

        __m512i acc0 = acc[0];
        for (int g = 1; g < NG; g++) { acc0 = _mm512_add_epi32(acc0, acc[g]); }

        // int correction + scale (src0 scales, slab_local)
        const int s_full = slab_local * NB + s;
        const __m512i scales_16 = _mm512_load_si512((const __m512i *) &src0.scales_t[s_full * TILED_MICRO + 0]);
        s1_acc = _mm512_add_epi32(s1_acc, _mm512_mullo_epi32(_mm512_sub_epi32(acc0, bias_16), scales_16));

        if constexpr (HAS_MIN) {
            // src1 bsums (absolute slab, 16-row table stride, col 0)
            int32_t sum = 0;
            for (int u = 0; u < NS; u++) { sum += src1.bsums[(s16_s1 + s * NS + u) * TILED_MICRO + 0]; }
            const __m512i mins_16 = _mm512_load_si512((const __m512i *) &src0.mins_t[s_full * TILED_MICRO + 0]);
            s2_acc = _mm512_add_epi32(s2_acc, _mm512_mullo_epi32(mins_16, _mm512_set1_epi32(sum)));
        }
    }

    // epilogue: accumulate into acc_f (no store to buf)
    const __m512 d0_vec = _mm512_load_ps(&src0.d[d0_base + 0]);
    __m512 f1 = _mm512_cvtepi32_ps(s1_acc);
    __m512 result = _mm512_mul_ps(f1, d0_vec);
    if constexpr (HAS_MIN) {
        const __m512 dmin_vec = _mm512_load_ps(&src0.dmin[d0_base + 0]);
        __m512 f2 = _mm512_cvtepi32_ps(s2_acc);
        result = _mm512_fnmadd_ps(dmin_vec, f2, result);
    }
    const float d1 = src1.d[d1_base + 0];
    *acc_f = _mm512_add_ps(*acc_f, _mm512_mul_ps(result, _mm512_set1_ps(d1)));
}

#endif // __AVX512VNNI__ && __AVX512VL__

#if defined(__AVX2__)

// AVX2 kernel: maddubs with the sign trick (vpsignb) on the true values.
// src1 is stored biased (+128); de-bias to true s1, then A = |s1| (u8) and
// B = q0 * sign(s1) (i8) so maddubs gives sum(s1 * q0) directly -- no 4-bit
// split, no bias correction. Max i16 pair = 2 * 128 * 127 = 32512 < 32767
// (holds for every type: |s1| <= 128, |q0| <= 127).
template <int SUBBLK, bool HAS_MIN, int BIAS>
static void tiled_run_microtile_avx2(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                     int i0, int j0, int n_cols, float * buf, int buf_stride) {
    constexpr int NB = TILED_TILE_K / SUBBLK;
    constexpr int NS = SUBBLK / 16;
    constexpr int GROUP = 8; // 8 src1 columns per group

    static_assert(SUBBLK == 16 || SUBBLK == 32, "unsupported SUBBLK");

    auto hsum256_epi32 = [](const __m256i v) -> int32_t {
        __m128i low = _mm256_castsi256_si128(v);
        __m128i high = _mm256_extracti128_si256(v, 1);
        __m128i sum128 = _mm_add_epi32(low, high);
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
        return _mm_cvtsi128_si32(sum128);
    };

    for (int i = 0; i < TILED_MICRO; i++) {
        const int ar = i0 + i;
        const float d0 = src0.d[ar];
        const float dmin0 = src0.dmin[ar];
        const uint8_t * q0 = (const uint8_t *) &src0.q[ar * TILED_TILE_K];
        const int32_t * scales_row = &src0.scales[ar * NB];
        const int32_t * mins_row = &src0.mins[ar * NB];

        for (int g = 0; g < TILED_MICRO; g += GROUP) {
            int tg = n_cols - g;
            if (tg <= 0) break;
            if (tg > GROUP) tg = GROUP;
            const uint8_t * q1_ptr[GROUP];
            for (int t = 0; t < GROUP; t++) {
                q1_ptr[t] = &src1.q[(j0 + g + t) * TILED_TILE_K];
            }

            __m256i s2   = _mm256_setzero_si256();
            int32_t bias_corr = 0;

            __m256i acc[GROUP];
            for (int t = 0; t < GROUP; t++) {
                acc[t] = _mm256_setzero_si256();
            }

            if constexpr (SUBBLK == 32) {
                for (int s = 0; s < NB; s++) {
                    const __m256i q0_32 = _mm256_load_si256((const __m256i *) &q0[s * SUBBLK]);
                    const __m256i scales16 = _mm256_set1_epi16(scales_row[s]);

                    // src1 bsums for 8 cols (per-column, for min correction)
                    const __m256i bsums_v = _mm256_add_epi32(
                        _mm256_load_si256((const __m256i *) &src1.bsums[s * NS * TILED_TILE_ROWS + j0 + g]),
                        _mm256_load_si256((const __m256i *) &src1.bsums[(s * NS + 1) * TILED_TILE_ROWS + j0 + g]));

                    if constexpr (BIAS == 0) {
                        // small codes (|q0| <= 31): one maddubs on the biased src1, corrected by
                        // 128*sum(q0); cheaper than the sign trick
                        int32_t bs0 = 0;
                        for (int u = 0; u < NS; u++) { bs0 += src0.bsums[(s * NS + u) * TILED_TILE_ROWS + ar]; }
                        bias_corr += 128 * bs0 * scales_row[s];
                        #pragma GCC unroll 8
                        for (int t = 0; t < tg; t++) {
                            const __m256i q1_32 = _mm256_load_si256((const __m256i *) &q1_ptr[t][s * SUBBLK]);
                            acc[t] = _mm256_add_epi32(acc[t],
                                _mm256_madd_epi16(scales16, _mm256_maddubs_epi16(q1_32, q0_32)));
                        }
                    } else {
                        // sign trick: de-bias src1 to true s1, A = |s1|, B = q0*sign(s1) -> sum(s1*q0)
                        #pragma GCC unroll 8
                        for (int t = 0; t < tg; t++) {
                            const __m256i q1_32 = _mm256_load_si256((const __m256i *) &q1_ptr[t][s * SUBBLK]);
                            const __m256i s1 = _mm256_xor_si256(q1_32, _mm256_set1_epi8((int8_t) 0x80));
                            const __m256i a_abs = _mm256_sign_epi8(s1, s1);
                            const __m256i b_sgn = _mm256_sign_epi8(q0_32, s1);
                            acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(scales16, _mm256_maddubs_epi16(a_abs, b_sgn)));
                        }
                    }

                    if constexpr (HAS_MIN) {
                        s2 = _mm256_add_epi32(s2, _mm256_mullo_epi32(bsums_v, _mm256_set1_epi32(mins_row[s])));
                    }
                }
            } else { // SUBBLK == 16
                for (int sp = 0; sp < NB; sp += 2) {
                    const __m256i q0_32 = _mm256_load_si256((const __m256i *) &q0[sp * SUBBLK]);
                    const __m256i scalesv = _mm256_set_m128i(_mm_set1_epi16(scales_row[sp + 1]), _mm_set1_epi16(scales_row[sp]));
                    const __m256i bsums0_v = _mm256_load_si256((const __m256i *) &src1.bsums[sp * TILED_TILE_ROWS + j0 + g]);
                    const __m256i bsums1_v = _mm256_load_si256((const __m256i *) &src1.bsums[(sp + 1) * TILED_TILE_ROWS + j0 + g]);

                    if constexpr (BIAS == 0) {
                        // small codes (|q0| <= 3): one maddubs on the biased src1, corrected by bias_corr
                        bias_corr += 128 * src0.bsums[sp * TILED_TILE_ROWS + ar] * scales_row[sp];
                        bias_corr += 128 * src0.bsums[(sp + 1) * TILED_TILE_ROWS + ar] * scales_row[sp + 1];
                        #pragma GCC unroll 8
                        for (int t = 0; t < tg; t++) {
                            const __m256i q1_32 = _mm256_load_si256((const __m256i *) &q1_ptr[t][sp * SUBBLK]);
                            acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(
                                scalesv, _mm256_maddubs_epi16(q1_32, q0_32)));
                        }
                    } else {
                        // sign trick: de-bias src1 to true s1, A = |s1|, B = q0*sign(s1) -> sum(s1*q0)
                        #pragma GCC unroll 8
                        for (int t = 0; t < tg; t++) {
                            const __m256i q1_32 = _mm256_load_si256((const __m256i *) &q1_ptr[t][sp * SUBBLK]);
                            const __m256i s1 = _mm256_xor_si256(q1_32, _mm256_set1_epi8((int8_t) 0x80));
                            const __m256i a_abs = _mm256_sign_epi8(s1, s1);
                            const __m256i b_sgn = _mm256_sign_epi8(q0_32, s1);
                            acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(scalesv, _mm256_maddubs_epi16(a_abs, b_sgn)));
                        }
                    }

                    if constexpr (HAS_MIN) {
                        s2 = _mm256_add_epi32(s2, _mm256_add_epi32(
                            _mm256_mullo_epi32(bsums0_v, _mm256_set1_epi32(mins_row[sp])),
                            _mm256_mullo_epi32(bsums1_v, _mm256_set1_epi32(mins_row[sp + 1]))));
                    }
                }
            }

            // epilogue
            alignas(32) int32_t s1_vals[GROUP];
            #pragma GCC unroll 8
            for (int t = 0; t < GROUP; t++) {
                s1_vals[t] = hsum256_epi32(acc[t]) - bias_corr;
            }
            __m256i s1_vec = _mm256_load_si256((const __m256i *) s1_vals);

            const __m256 src1_d_vec = _mm256_load_ps(&src1.d[j0 + g]);
            __m256 res_vec = _mm256_mul_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(s1_vec));

            if constexpr (HAS_MIN) {
#if defined(__FMA__)
                res_vec = _mm256_fnmadd_ps(_mm256_set1_ps(dmin0), _mm256_cvtepi32_ps(s2), res_vec);
#else
                res_vec = _mm256_sub_ps(res_vec, _mm256_mul_ps(_mm256_set1_ps(dmin0), _mm256_cvtepi32_ps(s2)));
#endif
            }

            // [col][row] buf: 8 cols for 1 row are not contiguous, store individually
            alignas(32) float res_vals[8];
            _mm256_storeu_ps(res_vals, _mm256_mul_ps(res_vec, src1_d_vec));
            #pragma GCC unroll 8
            for (int t = 0; t < tg; t++) {
                buf[(j0 + g + t) * buf_stride + ar] += res_vals[t];
            }
        }
    }
}

// sliding GEMV MAC (AVX2 body): the driver dequants the weight row into tiled_slide_w and
// the activation into tiled_slide_s1; this MACs the dequantized codes. maddubs per
// subblock (weight codes are the unsigned operand, activation the signed int8, so the
// product is exact - no +128 correction), scaled by the per-subblock weight scale, with the
// min term (weight min x per-32 activation code sum). Folded to float per 256-block (std's
// accumulation order). Reads the dequantized layout; no quant-type knowledge.
// hsum of the 8 int32 lanes of a 256-bit vector (the per-subblock maddubs scale sum)
static int32_t tiled_slide_hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64   = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64  = _mm_add_epi32(hi64, sum128);
    const __m128i hi32   = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

template <int SUBBLK, bool HAS_MIN>
static void tiled_run_gemv_int8_avx2(const tiled_slide_w & w, const tiled_slide_s1 & s1,
                                     float * out, int k) {
    constexpr int NB = TILED_TILE_K / SUBBLK;   // subblocks per 256-block (8 for SUBBLK=32)
    static_assert(SUBBLK == 32, "only SUBBLK=32 (q5_K) supported yet");
    static_assert(NB * SUBBLK == TILED_TILE_K, "SUBBLK must divide TILED_TILE_K");
    const int nb = k / TILED_TILE_K;
    const int8_t  * cw = w.code;
    const int8_t  * cs = s1.code;
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const int32_t * scb = w.scales + b * NB;
        const int32_t * mnb = w.mins   + b * NB;
        const int16_t * abb = s1.bsums32 + b * NB;
        const int8_t  * wcb = cw + b * TILED_TILE_K;
        const int8_t  * acb = cs + b * TILED_TILE_K;
        __m256i sumi = _mm256_setzero_si256();
        int32_t mt = 0;
        for (int s = 0; s < NB; s++) {
            const __m256i wc = _mm256_loadu_si256((const __m256i *) (wcb + s * SUBBLK));
            const __m256i ac = _mm256_loadu_si256((const __m256i *) (acb + s * SUBBLK));
            __m256i p = _mm256_maddubs_epi16(wc, ac);
            p = _mm256_madd_epi16(_mm256_set1_epi16((int16_t) scb[s]), p);
            sumi = _mm256_add_epi32(sumi, p);
            if constexpr (HAS_MIN) { mt += (int32_t) mnb[s] * (int32_t) abb[s]; }
        }
        acc += s1.d[b] * ( w.d[b] * (float) tiled_slide_hsum_i32_8(sumi)
                          - (HAS_MIN ? w.dmin[b] * (float) mt : 0.0f) );
    }
    *out = acc;
}
#endif // __AVX2__

#if defined(__AVX__) && !defined(__AVX2__)
template <int SUBBLK, bool HAS_MIN, int BIAS>
static void tiled_run_microtile_avx(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                    int i0, int j0, int n_cols, float * buf, int buf_stride) {
    constexpr int NB = TILED_TILE_K / SUBBLK;
    constexpr int NS = SUBBLK / 16;
    constexpr int GROUP = 4; // 4 src1 cols per group

    for (int i = 0; i < TILED_MICRO; i++) {
        const int ar = i0 + i;
        const float d0 = src0.d[ar];
        const float dmin0 = src0.dmin[ar];
        const uint8_t * q0 = (const uint8_t *) &src0.q[ar * TILED_TILE_K];
        const int32_t * scales_row = &src0.scales[ar * NB];
        const int32_t * mins_row = &src0.mins[ar * NB];

        for (int g = 0; g < TILED_MICRO; g += GROUP) {
            int tg = n_cols - g;
            if (tg <= 0) break;
            if (tg > GROUP) tg = GROUP;
            const uint8_t * q1g[GROUP];
            __m128i acc[GROUP];
            __m128i s2 = _mm_setzero_si128();
            int32_t bias_corr = 0;
            for (int t = 0; t < GROUP; t++) {
                q1g[t] = &src1.q[(j0 + g + t) * TILED_TILE_K];
                acc[t] = _mm_setzero_si128();
            }

            for (int s = 0; s < NB; s++) {
                const __m128i scales16 = _mm_set1_epi16(scales_row[s]);
                __m128i bsums_v = _mm_setzero_si128();
                for (int u = 0; u < NS; u++) {
                    const __m128i a16 = _mm_loadu_si128((const __m128i *) &q0[s * SUBBLK + u * 16]);
                    bsums_v = _mm_add_epi32(bsums_v, _mm_loadu_si128(
                        (const __m128i *) &src1.bsums[(s * NS + u) * TILED_TILE_ROWS + j0 + g]));
                    if constexpr (BIAS == 0) {
                        // small codes: one maddubs on the biased src1, corrected by bias_corr
                        for (int t = 0; t < tg; t++) {
                            const __m128i q1_16 = _mm_loadu_si128((const __m128i *) &q1g[t][s * SUBBLK + u * 16]);
                            acc[t] = _mm_add_epi32(acc[t], _mm_madd_epi16(scales16,
                                _mm_maddubs_epi16(q1_16, a16)));
                        }
                    } else {
                        // sign trick: de-bias src1 to true s1, A = |s1|, B = q0*sign(s1) -> sum(s1*q0)
                        for (int t = 0; t < tg; t++) {
                            const __m128i q1_16 = _mm_loadu_si128((const __m128i *) &q1g[t][s * SUBBLK + u * 16]);
                            const __m128i s1 = _mm_xor_si128(q1_16, _mm_set1_epi8((int8_t) 0x80));
                            const __m128i a_abs = _mm_sign_epi8(s1, s1);
                            const __m128i b_sgn = _mm_sign_epi8(a16, s1);
                            acc[t] = _mm_add_epi32(acc[t], _mm_madd_epi16(scales16, _mm_maddubs_epi16(a_abs, b_sgn)));
                        }
                    }
                }
                if constexpr (BIAS == 0) {
                    int32_t bs0 = 0;
                    for (int u = 0; u < NS; u++) { bs0 += src0.bsums[(s * NS + u) * TILED_TILE_ROWS + ar]; }
                    bias_corr += 128 * bs0 * scales_row[s];
                }
                if constexpr (HAS_MIN) {
                    s2 = _mm_add_epi32(s2, _mm_mullo_epi32(bsums_v, _mm_set1_epi32(mins_row[s])));
                }
            }

            int32_t s2_s[GROUP] = { 0, 0, 0, 0 };
            if constexpr (HAS_MIN) {
                _mm_storeu_si128((__m128i *) s2_s, s2);
            }

            for (int t = 0; t < tg; t++) {
                __m128i v = _mm_shuffle_epi32(acc[t], _MM_SHUFFLE(2, 3, 0, 1));
                acc[t] = _mm_add_epi32(acc[t], v);
                v = _mm_shuffle_epi32(acc[t], _MM_SHUFFLE(1, 0, 3, 2));
                acc[t] = _mm_add_epi32(acc[t], v);
                const int32_t s1 = _mm_cvtsi128_si32(acc[t]) - bias_corr;

                float res = d0 * (float) s1;
                if constexpr (HAS_MIN) {
                    res -= dmin0 * (float) s2_s[t];
                }
                buf[(j0 + g + t) * buf_stride + ar] += src1.d[j0 + g + t] * res;
            }
        }
    }
}

#endif // __AVX__ && !__AVX2__

// main microtile entry point
template <int SUBBLK, bool HAS_MIN, int BIAS>
void tiled_run_microtile(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                         int i0, int j0, int n_cols, float * buf, int buf_stride) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
    // normal 256-K slab geometry: k_extent = TILED_TILE_K, slab_offset = 0, the transposed
    // side-table stride is TILED_TILE_ROWS, and the acc row equals the tile row (i0_buf = i0)
    tiled_run_microtile_vnni<SUBBLK, HAS_MIN, BIAS>(src0, src1, i0, j0, n_cols, buf, buf_stride,
                                                    TILED_TILE_K, 0, TILED_TILE_ROWS, i0);
#elif defined(__AVX2__)
    tiled_run_microtile_avx2<SUBBLK, HAS_MIN, BIAS>(src0, src1, i0, j0, n_cols, buf, buf_stride);
#elif defined(__AVX__)
    tiled_run_microtile_avx<SUBBLK, HAS_MIN, BIAS>(src0, src1, i0, j0, n_cols, buf, buf_stride);
#else
    tiled_run_microtile_scalar<SUBBLK, HAS_MIN, BIAS>(src0, src1, i0, j0, n_cols, buf, buf_stride);
#endif
}

// extended-K microtile entry point (16 x K_full tile, one 256-K slab per call). VNNI only;
// the driver gates on tiled_kernel_ext_available() before calling.
template <int SUBBLK, bool HAS_MIN, int BIAS>
void tiled_run_microtile_ext(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                             int i0, int j0, int n_cols, float * buf, int buf_stride,
                             int k_extent, int slab_offset, int table_stride, int i0_buf) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
    tiled_run_microtile_vnni<SUBBLK, HAS_MIN, BIAS>(src0, src1, i0, j0, n_cols, buf, buf_stride,
                                                    k_extent, slab_offset, table_stride, i0_buf);
#else
    GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(i0); GGML_UNUSED(j0); GGML_UNUSED(n_cols);
    GGML_UNUSED(buf); GGML_UNUSED(buf_stride); GGML_UNUSED(k_extent); GGML_UNUSED(slab_offset);
    GGML_UNUSED(table_stride); GGML_UNUSED(i0_buf);
#endif
}

// GEMV (1 activation col x 16 weight rows) entry point on the extended tile. j0 is the col
// index (0 for the single real col). VNNI only; the driver gates on
// tiled_kernel_ext_available() before calling.
template <int SUBBLK, bool HAS_MIN, int BIAS>
void tiled_run_gemv_ext(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                        int j0, float * buf, int buf_stride,
                        int k_extent, int slab_offset, int table_stride, int i0_buf) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
    tiled_run_gemv_vnni<SUBBLK, HAS_MIN, BIAS>(src0, src1, j0, buf, buf_stride,
                                               k_extent, slab_offset, table_stride, i0_buf);
#else
    GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(j0); GGML_UNUSED(buf); GGML_UNUSED(buf_stride);
    GGML_UNUSED(k_extent); GGML_UNUSED(slab_offset); GGML_UNUSED(table_stride); GGML_UNUSED(i0_buf);
#endif
}

// L1-resident GEMV entry point (Driver C v2). VNNI only.
template <int SUBBLK, bool HAS_MIN, int BIAS>
void tiled_run_gemv_l1(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                       int slab_local, int slab_s1, void * acc_f,
                       int k_extent_s0, int k_extent_s1) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
    tiled_run_gemv_l1_vnni<SUBBLK, HAS_MIN, BIAS>(src0, src1, slab_local, slab_s1,
                                                  (__m512 *) acc_f, k_extent_s0, k_extent_s1);
#else
    GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(slab_local); GGML_UNUSED(slab_s1);
    GGML_UNUSED(acc_f); GGML_UNUSED(k_extent_s0); GGML_UNUSED(k_extent_s1);
#endif
}

// sliding GEMV MAC entry point. The driver dequants the weight row into w (decode_slide_src0)
// and the activation into s1 (decode_slide_src1), then calls this generic int8 MAC for each
// weight row. AVX2 only; the driver gates on tiled_kernel_slide_available() before calling.
template <int SUBBLK, bool HAS_MIN>
void tiled_run_gemv_int8(const tiled_slide_w & w, const tiled_slide_s1 & s1,
                         float * out, int k) {
#if defined(__AVX2__)
    tiled_run_gemv_int8_avx2<SUBBLK, HAS_MIN>(w, s1, out, k);
#else
    GGML_UNUSED(w); GGML_UNUSED(s1); GGML_UNUSED(out); GGML_UNUSED(k);
#endif
}

// explicit instantiations for the in-use formats (q4_K and q5_K share the constants)
template void tiled_run_microtile<32, true, 0>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                               int i0, int j0, int n_cols, float * buf, int buf_stride);
// iq4_xs and the other iq types: LUT-expanded codes, BIAS = 128
template void tiled_run_microtile<32, false, 128>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                                  int i0, int j0, int n_cols, float * buf, int buf_stride);
template void tiled_run_microtile<16, false, 128>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                                  int i0, int j0, int n_cols, float * buf, int buf_stride);
template void tiled_run_microtile<16, false, 32>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                                 int i0, int j0, int n_cols, float * buf, int buf_stride);
template void tiled_run_microtile<16, false, 4>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                                int i0, int j0, int n_cols, float * buf, int buf_stride);
template void tiled_run_microtile<16, true, 0>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                               int i0, int j0, int n_cols, float * buf, int buf_stride);

// extended-K entry point, same format set
#define TILED_EXT_INST(SB, HM, BI) \
    template void tiled_run_microtile_ext<SB, HM, BI>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1, \
        int i0, int j0, int n_cols, float * buf, int buf_stride, int k_extent, int slab_offset, int table_stride, int i0_buf);
TILED_EXT_INST(32, true , 0)
TILED_EXT_INST(32, false, 128)
TILED_EXT_INST(16, false, 128)
TILED_EXT_INST(16, false, 32)
TILED_EXT_INST(16, false, 4)
TILED_EXT_INST(16, true , 0)
#undef TILED_EXT_INST

// GEMV (1 col x 16 rows), same format set
#define TILED_GEMV_INST(SB, HM, BI) \
    template void tiled_run_gemv_ext<SB, HM, BI>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1, \
        int j0, float * buf, int buf_stride, int k_extent, int slab_offset, int table_stride, int i0_buf);
TILED_GEMV_INST(32, true , 0)
TILED_GEMV_INST(32, false, 128)
TILED_GEMV_INST(16, false, 128)
TILED_GEMV_INST(16, false, 32)
TILED_GEMV_INST(16, false, 4)
TILED_GEMV_INST(16, true , 0)
#undef TILED_GEMV_INST

// L1-resident GEMV (Driver C v2), same format set
#define TILED_GEMVL1_INST(SB, HM, BI) \
    template void tiled_run_gemv_l1<SB, HM, BI>(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1, \
        int slab_local, int slab_s1, void * acc_f, int k_extent_s0, int k_extent_s1);
TILED_GEMVL1_INST(32, true , 0)
TILED_GEMVL1_INST(32, false, 128)
TILED_GEMVL1_INST(16, false, 128)
TILED_GEMVL1_INST(16, false, 32)
TILED_GEMVL1_INST(16, false, 4)
TILED_GEMVL1_INST(16, true , 0)
#undef TILED_GEMVL1_INST


// sliding GEMV int8 MAC (q5_K: SUBBLK=32, HAS_MIN=true)
template void tiled_run_gemv_int8<32, true>(const tiled_slide_w & w, const tiled_slide_s1 & s1,
                                            float * out, int k);


#define MIN(a, b) ((a) < (b) ? (a) : (b))

// block_q8_K in 4-byte words, for the int32 gather indices
static_assert(sizeof(block_q8_K) == 292 && offsetof(block_q8_K, qs) == 4,
              "block_q8_K layout changed, fix the src1 repack");

// VNNI: interleave the natural [row][256] src0 codes in-place into group-local
// [kg][row][4] layout. Also transpose scales/mins to [s * ROWS + row] for
// the kernel's 512-bit per-lane correction loads.
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)

void tiled_repack_src0(tiled_tile_src0 * tile, int nb) {
    // transpose scales/mins to [s * ROWS + row]
    for (int s = 0; s < nb; s++) {
        for (int r = 0; r < TILED_TILE_ROWS; r++) {
            tile->scales_t[s * TILED_TILE_ROWS + r] = tile->scales[r * nb + s];
            tile->mins_t[s * TILED_TILE_ROWS + r] = tile->mins[r * nb + s];
        }
    }

    // interleave codes (in-place: each 16x16 int32 tile transposes onto itself)
    for (int grp = 0; grp < TILED_TILE_ROWS / TILED_MICRO; grp++) {
        uint8_t * base = (uint8_t *) &tile->q[grp * (TILED_MICRO * TILED_TILE_K)];
        for (int c = 0; c < 4; c++) {
            tiled_repack_16x16(base, c, TILED_TILE_K);
        }
    }
}
#else
void tiled_repack_src0(tiled_tile_src0 * tile, int nb) {
    GGML_UNUSED(tile);
    GGML_UNUSED(nb);
}
#endif

// per-group repack: transpose + interleave one 16-row group
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
void tiled_repack_src0_group(tiled_tile_src0 * tile, int grp, int nb) {
    const int r0 = grp * TILED_MICRO;
    // transpose scales/mins for this group
    for (int s = 0; s < nb; s++) {
        for (int r = r0; r < r0 + TILED_MICRO; r++) {
            tile->scales_t[s * TILED_TILE_ROWS + r] = tile->scales[r * nb + s];
            tile->mins_t[s * TILED_TILE_ROWS + r] = tile->mins[r * nb + s];
        }
    }
    // interleave codes for this group (in-place)
    uint8_t * base = (uint8_t *) &tile->q[r0 * TILED_TILE_K];
    for (int c = 0; c < 4; c++) {
        tiled_repack_16x16(base, c, TILED_TILE_K);
    }
}
#else
void tiled_repack_src0_group(tiled_tile_src0 * tile, int grp, int nb) {
    GGML_UNUSED(tile);
    GGML_UNUSED(grp);
    GGML_UNUSED(nb);
}
#endif

// extended repack: one 16-row group holding the full K (row stride k_extent). The scales/
// mins transpose to [s_full * TILED_MICRO + row] (the 16-row stride), the codes interleave
// over all k_extent/64 chunks. VNNI only (the extended path is VNNI-only).
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
void tiled_repack_src0_ext(tiled_tile_src0 * tile, int k_extent, int nb) {
    for (int s = 0; s < nb; s++) {
        for (int r = 0; r < TILED_MICRO; r++) {
            tile->scales_t[s * TILED_MICRO + r] = tile->scales[r * nb + s];
            tile->mins_t[s * TILED_MICRO + r] = tile->mins[r * nb + s];
        }
    }
    uint8_t * base = (uint8_t *) &tile->q[0];
    for (int c = 0; c < k_extent / (TILED_MICRO * 4); c++) {
        tiled_repack_16x16(base, c, k_extent);
    }
}
#else
void tiled_repack_src0_ext(tiled_tile_src0 * tile, int k_extent, int nb) {
    GGML_UNUSED(tile);
    GGML_UNUSED(k_extent);
    GGML_UNUSED(nb);
}
#endif

// In-place transpose of one 16-row x 64-k chunk of src0 codes.
// base points at the 16-row group (row r at base + r*256); row r is row-major.
// c selects the chunk (0..3): k in [c*64, c*64+64) = 16 int32 k-groups of 4 k-values.
// After the call, the 16x16 int32 tile [row][k-group] is transposed in place, so the
// kernel reads the 16 rows' k-group (kg%16) as one 512-bit vector at
// base + (kg%16)*256 + (kg/16)*64. All 16 loads retire before any store, so it is safe.
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
void tiled_repack_16x16(uint8_t * base, int c, int k_extent) {
    __m512i v[16];
    for (int r = 0; r < 16; r++) {
        v[r] = _mm512_load_si512((const __m512i *) (base + r * k_extent + c * (TILED_MICRO * 4)));
    }

    // 16x16 int32 transpose, 4 butterfly phases
    // Phase 1: 1-element interleave, pairs (0,1), (2,3), ..., (14,15)
    {
        const __m512i idx_a = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
        const __m512i idx_b = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
        for (int i = 0; i < 16; i += 2) {
            __m512i a = v[i], b = v[i+1];
            v[i]   = _mm512_permutex2var_epi32(a, idx_a, b);
            v[i+1] = _mm512_permutex2var_epi32(a, idx_b, b);
        }
    }
    // Phase 2: 2-element interleave, pairs (0,2), (1,3), (4,6), (5,7), ...
    {
        const __m512i idx_a = _mm512_setr_epi32(0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29);
        const __m512i idx_b = _mm512_setr_epi32(2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31);
        for (int i = 0; i < 16; i += 4) {
            __m512i a = v[i],   b = v[i+2];
            v[i]   = _mm512_permutex2var_epi32(a, idx_a, b);
            v[i+2] = _mm512_permutex2var_epi32(a, idx_b, b);
            a = v[i+1], b = v[i+3];
            v[i+1] = _mm512_permutex2var_epi32(a, idx_a, b);
            v[i+3] = _mm512_permutex2var_epi32(a, idx_b, b);
        }
    }
    // Phase 3: 4-element interleave, pairs (0,4), (1,5), ..., (7,11), (8,12), ...
    {
        const __m512i idx_a = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
        const __m512i idx_b = _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
        for (int i = 0; i < 16; i += 8) {
            for (int j = 0; j < 4; j++) {
                __m512i a = v[i+j], b = v[i+4+j];
                v[i+j]   = _mm512_permutex2var_epi32(a, idx_a, b);
                v[i+4+j] = _mm512_permutex2var_epi32(a, idx_b, b);
            }
        }
    }
    // Phase 4: 8-element interleave, pairs (0,8), (1,9), ..., (7,15)
    {
        const __m512i idx_a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
        const __m512i idx_b = _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
        for (int i = 0; i < 8; i++) {
            __m512i a = v[i], b = v[i+8];
            v[i]   = _mm512_permutex2var_epi32(a, idx_a, b);
            v[i+8] = _mm512_permutex2var_epi32(a, idx_b, b);
        }
    }

    // store: v[g] holds 16 int32s for k-group col_order[g], rows 0..15.
    // in-place: write back to the same chunk-c slice of row col_order[g]
    static const int col_order[16] = {0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15};
    for (int g = 0; g < 16; g++) {
        _mm512_store_si512((void *) (base + col_order[g] * k_extent + c * (TILED_MICRO * 4)), v[g]);
    }
}
#else
void tiled_repack_16x16(uint8_t * base, int c, int k_extent) {
    // scalar: transpose the 16x16 int32 tile (dead path on non-VNNI; kept correct via temp)
    alignas(64) uint8_t tmp[TILED_MICRO * TILED_MICRO * 4];
    for (int r = 0; r < 16; r++) {
        for (int g = 0; g < 16; g++) {
            const uint8_t * s = base + r * k_extent + c * (TILED_MICRO * 4) + g * 4;
            uint8_t * p = tmp + g * 64 + r * 4;
            memcpy(p, s, 4);
        }
    }
    for (int g = 0; g < 16; g++) {
        memcpy(base + g * k_extent + c * (TILED_MICRO * 4), tmp + g * 64, 64);
    }
}
#endif


