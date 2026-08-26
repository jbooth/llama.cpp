// Tiled matmul kernel: K-quant weights x q8_K activations.
// Per-ISA microtile branches: AVX512-VNNI (dpbusd), AVX2/AVX (maddubs,
// reference vec_dot style), scalar (SSE-only builds). The VNNI build also
// carries the src1 code interleave/unpack: the dpbusd kernel reads the src1
// codes in [k/4][row][4], so the driver's generic src1 unpack delegates to it.
// Templated on the src0 tile config (identical for q4_K/q5_K).

#include "tiled-kernel.h"
#include "ggml-cpu-impl.h"

#include <string.h>

#if defined(__AVX512VNNI__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

// block_q8_K in 4-byte words, for the int32 gather indices of the src1 code unpack
static constexpr int TILED_Q8_K_WORDS = sizeof(block_q8_K) / 4;
static_assert(sizeof(block_q8_K) == 292 && offsetof(block_q8_K, qs) == 4,
              "block_q8_K layout changed, fix the src1 gather indices");

static void tiled_interleave_src1_q8_K(const block_q8_K * rows, int64_t row_stride,
                                int64_t r_start, int64_t r_end,
                                int64_t n_k, int64_t nr1, int64_t nr1_pad, int8_t * qv) {
    // One-shot scatter of the src1 q8 codes of rows [r_start, r_end) into the
    // interleave region qv, laid out slab-major as [slab][k/4][row][4]:
    //
    //   qv[(s * TILED_TILE_K/4 + g) * nr1_pad + r] = qs[4*g + 0..3]
    //     of the src1 block (global row r, slab s)
    //
    // s in [0, n_slabs = n_k / TILED_TILE_K), g in [0, TILED_TILE_K/4),
    // r a GLOBAL row index in [0, nr1_pad); this call writes only
    // [r_start, r_end) (16-aligned) and rows points at global row r_start
    // (row r at rows + (r - r_start) * row_stride + s blocks). Rows r >= nr1
    // write zeros, so the 16-row padded tail is covered here (no separate
    // memset). One code byte per element; d and bsums stay in the natural
    // rows and are never touched. The [row][4] tail makes the (window,
    // slab) tile column 4B-contiguous per row, so tiled_unpack_src1_q8_K_kernel
    // copies it straight into the tile.
    // n_k is the region width in elements (already slab-padded by the
    // caller); the last slab may be partial and its excess groups read the
    // row's tail
    const int64_t n_slabs = n_k / TILED_TILE_K;
    GGML_ASSERT(qv);
    GGML_ASSERT(n_k % TILED_TILE_K == 0);
    GGML_ASSERT((r_start & 15) == 0 && (r_end & 15) == 0 && r_end <= nr1_pad);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
    // VNNI build: one masked 16-lane int32 gather + one 64B store per
    // (slab, k-group, 16-row) block; masked lanes (past nr1) gather zeros
    {
        // row_off[r] = int32 index (in words) of rows[r].qs[0] (d is 4B, qs at +4);
        // row_stride * TILED_Q8_K_WORDS stays in int32 for any realistic row count
        const int32_t row_stride_w = (int32_t) row_stride * TILED_Q8_K_WORDS;
        int32_t row_off[TILED_MICRO];
        for (int r = 0; r < TILED_MICRO; r++) row_off[r] = r * row_stride_w + 1;
        const __m512i idx_row = _mm512_loadu_si512((const __m512i *) row_off);
        const int32_t * base = (const int32_t *) rows;
        for (int64_t r0 = r_start; r0 < r_end; r0 += TILED_MICRO) {
            // masked lanes (past nr1) gather into the zero register, so the
            // 16-row pad tail of the region is zeroed here, no separate memset
            const int64_t n = (nr1 > r0) ? nr1 - r0 : 0;
            const __mmask16 k = (n >= TILED_MICRO) ? 0xffff : ((__mmask16) ((1u << n) - 1));
            const int64_t rl = r0 - r_start; // rows points at global row r_start
            for (int64_t s = 0; s < n_slabs; s++) {
                // block (row rl, slab s) is rl*row_stride + s blocks from rows
                const __m512i base_idx = _mm512_set1_epi32((int32_t) ((rl * row_stride + s) * TILED_Q8_K_WORDS));
                int8_t * out = qv + s * TILED_TILE_K * nr1_pad + r0 * 4;
                for (int g = 0; g < TILED_TILE_K / 4; g++) {
                    const __m512i idx = _mm512_add_epi32(_mm512_add_epi32(idx_row, base_idx), _mm512_set1_epi32(g));
                    const __m512i v = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), k, idx, base, 4);
                    _mm512_storeu_si512((void *) (out + g * nr1_pad * 4), v);
                }
            }
        }
        return;
    }
#endif
    // non-VNNI build: same region layout (the driver only calls this on VNNI builds)
    for (int64_t r0 = r_start; r0 < r_end; r0 += TILED_MICRO) {
        for (int64_t r = r0; r < r0 + TILED_MICRO; r++) {
            for (int64_t s = 0; s < n_slabs; s++) {
                int8_t * out = qv + (s * TILED_TILE_K) * nr1_pad + r * 4;
                if (r < nr1) {
                    const block_q8_K & x = rows[(r - r_start) * row_stride + s];
                    for (int g = 0; g < TILED_TILE_K / 4; g++) {
                        memcpy(out + g * nr1_pad * 4, x.qs + g * 4, 4);
                    }
                } else {
                    for (int g = 0; g < TILED_TILE_K / 4; g++) {
                        memset(out + g * nr1_pad * 4, 0, 4);
                    }
                }
            }
        }
    }
}

#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
// The dpbusd body reads the src1 codes in the [k/4][row][4] order (one 64B
// vector per k-group x 16 rows), so the driver's generic per-row memcpy
// cannot fill the tile; this op does (the driver calls it under
// KERNEL_SRC1_UNPACK). The tensor was interleaved up front by
// tiled_interleave_src1_q8_K, so the tile column is 4 bytes per row,
// contiguous. The window's 16-padded span [r_start, r_start + n_padded) can
// reach past the region's 16-padded row count for a ragged last window, so
// clamp the copy to the region and zero the tail: those rows are all
// >= n_rows (dropped at the store) and read the zeroed pad tail either way.
void tiled_unpack_src1_q8_K_kernel(int n_rows, tiled_tile_src1 * tile,
                                   const int8_t * qv, int64_t nr1_pad, int64_t r_start, int64_t kblk) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    const int n_padded = (n_rows + TILED_MICRO - 1) & ~(TILED_MICRO - 1);
    const int n_qv = TILED_TILE_K / 4;
    GGML_ASSERT(qv);
    const int n_copy = (int) MIN((int64_t) n_padded, nr1_pad - r_start);
    for (int g = 0; g < n_qv; g++) {
        memcpy(&tile->q[g * TILED_TILE_ROWS * 4],
               qv + ((int64_t) (kblk * n_qv + g) * nr1_pad + r_start) * 4, (size_t) n_copy * 4);
        memset(&tile->q[g * TILED_TILE_ROWS * 4 + (size_t) n_copy * 4], 0, (size_t) (n_padded - n_copy) * 4);
    }
}
#endif

// One-shot, all-threads preparation of the src1 interleave region (layout in
// tiled_interleave_src1_q8_K): the q8 codes of the whole tensor are scattered
// into a flat [slab][k/4-in-slab][row][4] wdata region (rows padded to 16,
// zeroed tail) so the per (window, slab) unpack becomes a contiguous copy.
// The 16-row group partition spans all threads; the F32 to q8_K conversion
// (split by k-block) must finish before any thread interleaves (split by row),
// hence the inner barrier. No-op on non-VNNI builds: the base is null and
// the unpack copies the natural q8 codes per call.
tiled_src1_interleave tiled_prepare_src1_interleave(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src1,
        enum ggml_type vec_dot_type,
        int64_t ne10,
        int64_t nr1,
        int ith,
        int nth) {
    tiled_src1_interleave res = { nullptr, 0 };
#if !defined(__AVX512VNNI__) || !defined(__AVX512VL__) || !defined(__AVX512DQ__)
    (void) params;
    (void) src1;
    (void) vec_dot_type;
    (void) ne10;
    (void) nr1;
    (void) ith;
    (void) nth;
    return res;
#endif
    res.nr1_pad = (nr1 + 15) & ~15LL;
    const int64_t k1_pad = (ne10 + 255) & ~255LL; // the region holds whole slabs
    int8_t * qv;
    const block_q8_K * src1_codes;
    int64_t src1_stride;
    if (src1->type != vec_dot_type) {
        ggml_barrier(params->threadpool);
        const size_t off = ggml_row_size(vec_dot_type, ne10) * (size_t) nr1;
        GGML_ASSERT(params->wsize >= off + (size_t) k1_pad * res.nr1_pad);
        qv = (int8_t *) ((char *) params->wdata + off);
        src1_codes = (const block_q8_K *) params->wdata;
        src1_stride = ne10 / 256; // wdata rows are contiguous 256 blocks
    } else {
        GGML_ASSERT(params->wsize >= (size_t) k1_pad * res.nr1_pad);
        qv = (int8_t *) params->wdata;
        src1_codes = (const block_q8_K *) src1->data;
        src1_stride = src1->nb[1] / sizeof(block_q8_K);
    }
    const int64_t n_groups = res.nr1_pad / 16;
    const int64_t g0 = (int64_t) ith * n_groups / nth;
    const int64_t g1 = (int64_t) (ith + 1) * n_groups / nth;
    for (int64_t g = g0; g < g1; g++) {
        tiled_interleave_src1_q8_K(src1_codes + g * 16 * src1_stride, src1_stride,
                                   g * 16, (g + 1) * 16, k1_pad, nr1, res.nr1_pad, qv);
    }
    res.qv = qv;
    return res;
}

template <typename T>
static void tiled_run_microtile_scalar(const T & src0, const tiled_tile_src1 & src1,
                                       int i0, int j0, float * buf, int buf_stride) {
    constexpr int NB = T::NB;    // subblocks per 256-K block
    constexpr int SUBBLK = TILED_TILE_K / NB;
    constexpr int NS = SUBBLK / 16; // per-16 bsums per subblock
    constexpr bool HAS_MIN = T::HAS_MIN_V;
    constexpr int BIAS = T::BIAS_V;

    float acc[TILED_MICRO][TILED_MICRO];
    memset(acc, 0, sizeof(acc));

    // subdots at subblock granularity over the 256-K block, exact integer math
    for (int s = 0; s < NB; s++) {
        for (int j = 0; j < TILED_MICRO; j++) {
            const int br = j0 + j;
            const int8_t * q1 = &src1.q[br * TILED_TILE_K + s * SUBBLK];
            int32_t bsum = 0;
            for (int u = 0; u < NS; u++) {
                bsum += src1.bsums[(s * NS + u) * TILED_TILE_ROWS + br];
            }

            for (int i = 0; i < TILED_MICRO; i++) {
                const int ar = i0 + i;
                const uint8_t * q0 = &src0.q[ar * TILED_TILE_K + s * SUBBLK];

                int32_t raw = 0;
                for (int e = 0; e < SUBBLK; e++) {
                    raw += (int32_t) q0[e] * (int32_t) q1[e];
                }

                // BIAS: subtract BIAS*bsum (src1's per-subblock code sum) from the
                // exact int raw (q3_K=4, q6_K=32; 0 otherwise, vanishes via constexpr)
                int32_t corr = raw;
                if constexpr (BIAS != 0) {
                    corr -= BIAS * bsum;
                }
                const int32_t scales_raw = (int32_t) src0.scales[ar * NB + s] * corr;
                // d1 is NOT applied here: it is constant over the s-loop, so
                // acc holds the d1-un-scaled sum and the store below applies
                // d1 once per element (same factoring as the VNNI kernel)
                if constexpr (HAS_MIN) {
                    const int32_t mins_bsum = (int32_t) src0.mins[ar * NB + s] * bsum;
                    acc[i][j] += (float) src0.d[ar] * (float) scales_raw
                              - (float) src0.dmin[ar] * (float) mins_bsum;
                } else {
                    acc[i][j] += (float) src0.d[ar] * (float) scales_raw;
                }
            }
        }
    }

    // accumulate the microtile into the j-major buffer (unconditional 16x16; rows/cols
    // past the window hold harmless tile garbage, dropped by the store); apply the
    // column's d1 here (hoisted out of the s-loop)
    for (int i = 0; i < TILED_MICRO; i++) {
        for (int j = 0; j < TILED_MICRO; j++) {
            buf[(i0 + i) * buf_stride + (j0 + j)] += src1.d[j0 + j] * acc[i][j];
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
// 8-row band (s2_acc has no dpbusd dependency, so it is computed after the
// band pass and adds no register pressure); a wider band spills.
template <typename T>
static void tiled_run_micro_vnni_int_8x16(const T & src0, const tiled_tile_src1 & src1,
                                          int i0, int j0, float * buf, int buf_stride) {
    constexpr int NB = T::NB;
    constexpr int SUBBLK = TILED_TILE_K / NB;
    constexpr int NS = SUBBLK / 16;
    constexpr int NG = SUBBLK / 4;
    constexpr bool HAS_MIN = T::HAS_MIN_V;
    constexpr int BIAS = T::BIAS_V;

    // band width, see the register-pressure note above
    constexpr int NUM_ROWS = 8;

    const __m512 d1_vec = _mm512_loadu_ps(&src1.d[j0]);

    __m512i s1_acc[NUM_ROWS];
    for (int t = 0; t < NUM_ROWS; t++) { s1_acc[t] = _mm512_setzero_si512(); }
    __m512i s2_acc[NUM_ROWS];
    if constexpr (HAS_MIN) {
        for (int t = 0; t < NUM_ROWS; t++) { s2_acc[t] = _mm512_setzero_si512(); }
    }

    const int32_t * bsums = src1.bsums; // int32 per-16 sums; NS > 1 combines NS lanes per subblock
    for (int s = 0; s < NB; s++) {
        __m512i bsums32 = _mm512_loadu_si512((const __m512i *) &bsums[s * NS * TILED_TILE_ROWS + j0]);
        for (int u = 1; u < NS; u++) {
            bsums32 = _mm512_add_epi32(bsums32, _mm512_loadu_si512((const __m512i *) &bsums[(s * NS + u) * TILED_TILE_ROWS + j0]));
        }

        __m512i bias32 = _mm512_setzero_si512();
        if constexpr (BIAS != 0) {
            bias32 = _mm512_mullo_epi32(bsums32, _mm512_set1_epi32(BIAS));
        }

        __m512i acc16[NUM_ROWS];
        for (int t = 0; t < NUM_ROWS; t++) { acc16[t] = _mm512_setzero_si512(); }

        #pragma GCC unroll 8
        for (int g = 0; g < NG; g++) {
            const int kg = s * NG + g;
            const __m512i codes = _mm512_loadu_si512((const __m512i *) &src1.q[kg * TILED_TILE_ROWS * 4 + j0 * 4]);
            for (int t = 0; t < NUM_ROWS; t++) {
                const uint32_t u4 = *(const uint32_t *) &src0.q[(i0 + t) * TILED_TILE_K + kg * 4];
                const __m512i u4b = _mm512_set1_epi32((int) u4);
                acc16[t] = _mm512_dpbusd_epi32(acc16[t], u4b, codes);
            }
        }

        // int correction: s1_acc += scales*(raw-BIAS*bsums)
        for (int t = 0; t < NUM_ROWS; t++) {
            const int ar = i0 + t;
            __m512i rawi = acc16[t];
            if constexpr (BIAS != 0) {
                rawi = _mm512_sub_epi32(rawi, bias32);
            }
            s1_acc[t] = _mm512_add_epi32(s1_acc[t],
                _mm512_mullo_epi32(rawi, _mm512_set1_epi32(src0.scales[ar * NB + s])));
        }
    }

    // s2_acc += mins*bsums; independent of the dpbusd results, so it runs here instead of in the
    // band pass: the band pass keeps the register budget for the 8-row band
    if constexpr (HAS_MIN) {
        for (int s = 0; s < NB; s++) {
            __m512i bsums32 = _mm512_loadu_si512((const __m512i *) &bsums[s * NS * TILED_TILE_ROWS + j0]);
            for (int u = 1; u < NS; u++) {
                bsums32 = _mm512_add_epi32(bsums32, _mm512_loadu_si512((const __m512i *) &bsums[(s * NS + u) * TILED_TILE_ROWS + j0]));
            }
            for (int t = 0; t < NUM_ROWS; t++) {
                s2_acc[t] = _mm512_add_epi32(s2_acc[t],
                    _mm512_mullo_epi32(bsums32, _mm512_set1_epi32(src0.mins[(i0 + t) * NB + s])));
            }
        }
    }

    // epilogue: int->float, apply per-row scales, store to buf
    for (int t = 0; t < NUM_ROWS; t++) {
        const int ar = i0 + t;
        __m512 f1 = _mm512_cvtepi32_ps(s1_acc[t]);
        __m512 result = _mm512_mul_ps(f1, _mm512_set1_ps(src0.d[ar]));
        if constexpr (HAS_MIN) {
            __m512 f2 = _mm512_cvtepi32_ps(s2_acc[t]);
            result = _mm512_fnmadd_ps(_mm512_set1_ps(src0.dmin[ar]), f2, result);
        }
        float * p = &buf[(i0 + t) * buf_stride + j0];
        _mm512_storeu_ps(p, _mm512_add_ps(_mm512_loadu_ps(p), _mm512_mul_ps(result, d1_vec)));
    }
}

// 16x16 microtile as two explicit 8x16 band passes
template <typename T>
static void tiled_run_microtile_vnni_int(const T & src0, const tiled_tile_src1 & src1,
                                         int i0, int j0, float * buf, int buf_stride) {
    tiled_run_micro_vnni_int_8x16(src0, src1, i0,      j0, buf, buf_stride);
    tiled_run_micro_vnni_int_8x16(src0, src1, i0 + 8,  j0, buf, buf_stride);
}

#endif // __AVX512VNNI__ && __AVX512VL__

#if defined(__AVX2__)

// =====================================================================
// AVX2 microtile kernel (maddubs; same algebra as the VNNI kernel)
// =====================================================================
// No outer-product dot exists here: _mm256_maddubs_epi16 pairs the lane
// elements of the two operands, so one instruction yields the 16 i16
// partial sums (two k products each) of ONE (src0-row, src1-col) pair, and
// the per-pair accumulator cannot extend across src0 rows. The j window
// is therefore processed in groups of GROUP columns: one i32
// accumulator per column, live across the subblock loop. GROUP = 8 gives
// 8 independent dot chains per (i, subblock) to hide the maddubs
// latency (the VNNI 8-row band analog); a few of the 8 accs spill to stack at
// the 16 YMM limit. The src0 subblock codes are loaded once per
// (i, s, group) and reused over the group (1/8 of a pure per-pair loop's
// src0-side load traffic). The 16x16 microtile only fixes the buf layout.
//
// SUBBLK=32: one 256-bit maddubs per (s, column), a 256-bit set1 scale,
// one 8-lane i32 accumulator per column.
// SUBBLK=16: two subblocks per 32B load. The 256-bit maddubs product
// still gives 16 i16 lanes (0..7 = subblock sp, 8..15 = sp+1), but the
// scale is applied per 128-bit half with a set1 (not one interleaved
// 8-scalar-arg set that the compiler lowers through a long FPU chain
// and stack sign-extends). Each column therefore holds two 4-lane i32
// accumulators (lo/hi), reduced together in the epilogue.
//
// Math: the same int split-pass as the VNNI kernel. Per (i, j) the
// accumulators hold scales*raw over the subblocks, so the -> scalar reduce
// happens ONCE per pair (in the group epilogue). The per-subblock BIAS
// and mins corrections run as 256-bit i32 vectors, one lane per src1 column,
// with bsums from the per-16 bsum table. s1 = hsum(acc) - BIAS*sum
// (scales*bsums), s2 = sum(mins*bsums); the epilogue is scalar: buf += d1 * (d0*s1
// - dmin*s2). No FMA is used (the epilogue is scalar), so the AVX2 tier
// alone is sufficient.

template <typename T>
static void tiled_run_microtile_avx2(const T & src0, const tiled_tile_src1 & src1,
                                     int i0, int j0, float * buf, int buf_stride) {
    constexpr int NB = T::NB;
    constexpr int SUBBLK = TILED_TILE_K / NB;
    constexpr bool HAS_MIN = T::HAS_MIN_V;
    constexpr int BIAS = T::BIAS_V;
    constexpr int GROUP = 8; // src1 columns per group: one acc32 per column

    static_assert(SUBBLK == 16 || SUBBLK == 32, "unsupported SUBBLK");

    for (int i = 0; i < TILED_MICRO; i++) {
        // 4-lane i32 hsum (used in the group epilogue)
        auto hsum4 = [](const __m128i v) {
            __m128i s = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtsi128_si32(s);
        };
        const int ar = i0 + i;
        const float d0 = src0.d[ar];
        const float dmin0 = src0.dmin[ar];
        const uint8_t * q0 = &src0.q[ar * TILED_TILE_K];
        const int32_t * scales_row = &src0.scales[ar * NB];
        const int32_t * mins_row = &src0.mins[ar * NB];

        for (int g = 0; g < TILED_MICRO; g += GROUP) {
            const int8_t * q1g[GROUP];
            __m256i corr = _mm256_setzero_si256(); // -BIAS * sum_s scales_s * bsums_s, 1 lane per col (vanishes via constexpr)
            __m256i s2 = _mm256_setzero_si256();   // sum_s mins_s * bsums_s
            if constexpr (SUBBLK == 32) {
                __m256i acc[GROUP];
                for (int t = 0; t < GROUP; t++) {
                    q1g[t] = &src1.q[(j0 + g + t) * TILED_TILE_K];
                    acc[t] = _mm256_setzero_si256();
                }
                // one 32B a load and one 256-bit maddubs per (s, column)
                for (int s = 0; s < NB; s++) {
                    const __m256i q0_32 = _mm256_loadu_si256((const __m256i *) &q0[s * SUBBLK]);
                    const __m256i scales16 = _mm256_set1_epi16(scales_row[s]);
                    const __m256i bsums_v = _mm256_add_epi32(
                        _mm256_loadu_si256((const __m256i *) &src1.bsums[s * 2 * TILED_TILE_ROWS + j0 + g]),
                        _mm256_loadu_si256((const __m256i *) &src1.bsums[(s * 2 + 1) * TILED_TILE_ROWS + j0 + g]));
                    for (int t = 0; t < GROUP; t++) {
                        acc[t] = _mm256_add_epi32(acc[t],
                            _mm256_madd_epi16(scales16, _mm256_maddubs_epi16(
                                q0_32, _mm256_loadu_si256((const __m256i *) &q1g[t][s * SUBBLK]))));
                    }
                    if constexpr (BIAS != 0) {
                        corr = _mm256_sub_epi32(corr, _mm256_mullo_epi32(bsums_v, _mm256_set1_epi32(BIAS * scales_row[s])));
                    }
                    if constexpr (HAS_MIN) {
                        s2 = _mm256_add_epi32(s2, _mm256_mullo_epi32(bsums_v, _mm256_set1_epi32(mins_row[s])));
                    }
                }
                int32_t corr_s[GROUP] = { 0 };
                int32_t s2_s[GROUP] = { 0 };
                if constexpr (BIAS != 0) {
                    _mm256_storeu_si256((__m256i *) corr_s, corr);
                }
                if constexpr (HAS_MIN) {
                    _mm256_storeu_si256((__m256i *) s2_s, s2);
                }
                for (int t = 0; t < GROUP; t++) {
                    // 8 i32 lanes -> scalar (the per-pair dot, pre-correction)
                    const __m128i lo = _mm256_castsi256_si128(acc[t]);
                    const int32_t s1 = hsum4(_mm_add_epi32(lo, _mm256_extracti128_si256(acc[t], 1))) + corr_s[t];
                    float res = d0 * (float) s1;
                    if constexpr (HAS_MIN) {
                        res -= dmin0 * (float) s2_s[t];
                    }
                    buf[ar * buf_stride + j0 + g + t] += src1.d[j0 + g + t] * res;
                }
            } else {
                __m256i acc[GROUP];
                for (int t = 0; t < GROUP; t++) {
                    q1g[t] = &src1.q[(j0 + g + t) * TILED_TILE_K];
                    acc[t] = _mm256_setzero_si256();
                }
                // SUBBLK = 16: two subblocks per 32B load; i16 lanes 0..7 =
                // subblock sp, 8..15 = sp+1. Build the interleaved scale from
                // two 128-bit set1s (one set_m128i) instead of one 8-scalar-arg
                // 256-bit set, which the compiler lowers through a long FPU build
                // chain plus stack sign-extends.
                for (int sp = 0; sp < NB; sp += 2) {
                    const __m256i q0_32 = _mm256_loadu_si256((const __m256i *) &q0[sp * SUBBLK]);
                    const __m256i scalesv = _mm256_set_m128i(_mm_set1_epi16(scales_row[sp + 1]), _mm_set1_epi16(scales_row[sp]));
                    const __m256i bsums0_v = _mm256_loadu_si256((const __m256i *) &src1.bsums[sp * TILED_TILE_ROWS + j0 + g]);
                    const __m256i bsums1_v = _mm256_loadu_si256((const __m256i *) &src1.bsums[(sp + 1) * TILED_TILE_ROWS + j0 + g]);
                    for (int t = 0; t < GROUP; t++) {
                        acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(
                            scalesv, _mm256_maddubs_epi16(
                                q0_32, _mm256_loadu_si256((const __m256i *) &q1g[t][sp * SUBBLK]))));
                    }
                    // per-subblock scales differ, so corr/s2 keep the bsums0/bsums1 split
                    if constexpr (BIAS != 0) {
                        corr = _mm256_sub_epi32(corr, _mm256_add_epi32(
                            _mm256_mullo_epi32(bsums0_v, _mm256_set1_epi32(BIAS * scales_row[sp])),
                            _mm256_mullo_epi32(bsums1_v, _mm256_set1_epi32(BIAS * scales_row[sp + 1]))));
                    }
                    if constexpr (HAS_MIN) {
                        s2 = _mm256_add_epi32(s2, _mm256_add_epi32(
                            _mm256_mullo_epi32(bsums0_v, _mm256_set1_epi32(mins_row[sp])),
                            _mm256_mullo_epi32(bsums1_v, _mm256_set1_epi32(mins_row[sp + 1]))));
                    }
                }
                int32_t corr_s[GROUP] = { 0 };
                int32_t s2_s[GROUP] = { 0 };
                if constexpr (BIAS != 0) {
                    _mm256_storeu_si256((__m256i *) corr_s, corr);
                }
                if constexpr (HAS_MIN) {
                    _mm256_storeu_si256((__m256i *) s2_s, s2);
                }
                for (int t = 0; t < GROUP; t++) {
                    // 8 i32 lanes -> scalar (the per-pair dot, pre-correction)
                    const __m128i lo = _mm256_castsi256_si128(acc[t]);
                    const int32_t s1 = hsum4(_mm_add_epi32(lo, _mm256_extracti128_si256(acc[t], 1))) + corr_s[t];
                    float res = d0 * (float) s1;
                    if constexpr (HAS_MIN) {
                        res -= dmin0 * (float) s2_s[t];
                    }
                    buf[ar * buf_stride + j0 + g + t] += src1.d[j0 + g + t] * res;
                }
            }
        }
    }
}

#endif // __AVX2__

#if defined(__AVX__) && !defined(__AVX2__)

// =====================================================================
// AVX (128-bit) microtile kernel: the AVX2 math on 16 k codes per op
// =====================================================================
// _mm_maddubs_epi16 is 128-bit only on this tier (256-bit integer ops
// are AVX2), so each op covers one per-16 group: 16B loads, 8 i16
// partials, and a 4-lane i32 accumulator. Same group-of-4 structure as
// the AVX2 kernel (one 4-lane acc per src1 column across the subblock
// loop, src0 codes hoisted per group). GROUP = 8 was tried here and
// rejected: the 128-bit dot is maddubs + madd_epi16 per 16-k chunk
// (NS = 2 chunks for SUBBLK = 32), so 8 accs plus the dot temps spill
// 7 of the 8 accs to stack, regressing every format by 6-19% (4096^3).
// GROUP = 4 keeps all 4 accs in registers.

template <typename T>
static void tiled_run_microtile_avx(const T & src0, const tiled_tile_src1 & src1,
                                    int i0, int j0, float * buf, int buf_stride) {
    constexpr int NB = T::NB;
    constexpr int SUBBLK = TILED_TILE_K / NB;
    constexpr int NS = SUBBLK / 16;
    constexpr bool HAS_MIN = T::HAS_MIN_V;
    constexpr int BIAS = T::BIAS_V;
    constexpr int GROUP = 4; // src1 columns per group: one acc32 per column

    for (int i = 0; i < TILED_MICRO; i++) {
        const int ar = i0 + i;
        const float d0 = src0.d[ar];
        const float dmin0 = src0.dmin[ar];
        const uint8_t * q0 = &src0.q[ar * TILED_TILE_K];
        const int32_t * scales_row = &src0.scales[ar * NB];
        const int32_t * mins_row = &src0.mins[ar * NB];

        for (int g = 0; g < TILED_MICRO; g += GROUP) {
            const int8_t * q1g[GROUP];
            __m128i acc[GROUP];
            __m128i corr = _mm_setzero_si128(); // -BIAS * sum_s scales_s * bsums_s, 1 lane per col (vanishes via constexpr)
            __m128i s2 = _mm_setzero_si128();   // sum_s mins_s * bsums_s
            for (int t = 0; t < GROUP; t++) {
                q1g[t] = &src1.q[(j0 + g + t) * TILED_TILE_K];
                acc[t] = _mm_setzero_si128();
            }

            for (int s = 0; s < NB; s++) {
                const __m128i scales16 = _mm_set1_epi16(scales_row[s]);
                __m128i bsums_v = _mm_setzero_si128(); // per-col bsums as a 4-lane i32 vector
                for (int u = 0; u < NS; u++) {
                    const __m128i a16 = _mm_loadu_si128((const __m128i *) &q0[s * SUBBLK + u * 16]);
                    bsums_v = _mm_add_epi32(bsums_v, _mm_loadu_si128(
                        (const __m128i *) &src1.bsums[(s * NS + u) * TILED_TILE_ROWS + j0 + g]));
                    for (int t = 0; t < GROUP; t++) {
                        acc[t] = _mm_add_epi32(acc[t], _mm_madd_epi16(scales16,
                            _mm_maddubs_epi16(a16,
                                _mm_loadu_si128((const __m128i *) &q1g[t][s * SUBBLK + u * 16]))));
                    }
                }
                if constexpr (BIAS != 0) {
                    corr = _mm_sub_epi32(corr, _mm_mullo_epi32(bsums_v, _mm_set1_epi32(BIAS * scales_row[s])));
                }
                if constexpr (HAS_MIN) {
                    s2 = _mm_add_epi32(s2, _mm_mullo_epi32(bsums_v, _mm_set1_epi32(mins_row[s])));
                }
            }

            int32_t corr_s[GROUP] = { 0, 0, 0, 0 };
            int32_t s2_s[GROUP] = { 0, 0, 0, 0 };
            if constexpr (BIAS != 0) {
                _mm_storeu_si128((__m128i *) corr_s, corr);
            }
            if constexpr (HAS_MIN) {
                _mm_storeu_si128((__m128i *) s2_s, s2);
            }

            for (int t = 0; t < GROUP; t++) {
                // 4 i32 lanes -> scalar (the per-pair dot, pre-correction)
                __m128i v = _mm_shuffle_epi32(acc[t], _MM_SHUFFLE(2, 3, 0, 1));
                acc[t] = _mm_add_epi32(acc[t], v);
                v = _mm_shuffle_epi32(acc[t], _MM_SHUFFLE(1, 0, 3, 2));
                acc[t] = _mm_add_epi32(acc[t], v);
                const int32_t s1 = _mm_cvtsi128_si32(acc[t]) + corr_s[t];

                float res = d0 * (float) s1;
                if constexpr (HAS_MIN) {
                    res -= dmin0 * (float) s2_s[t];
                }
                buf[ar * buf_stride + j0 + g + t] += src1.d[j0 + g + t] * res;
            }
        }
    }
}

#endif // __AVX__ && !__AVX2__

template <typename T>
void tiled_run_microtile(const T & src0, const tiled_tile_src1 & src1,
                         int i0, int j0, float * buf, int buf_stride) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
    tiled_run_microtile_vnni_int(src0, src1, i0, j0, buf, buf_stride);
#elif defined(__AVX2__)
    tiled_run_microtile_avx2(src0, src1, i0, j0, buf, buf_stride);
#elif defined(__AVX__)
    tiled_run_microtile_avx(src0, src1, i0, j0, buf, buf_stride);
#else
    tiled_run_microtile_scalar(src0, src1, i0, j0, buf, buf_stride);
#endif
}

// explicit instantiations for the in-use tile types (q4_K and q5_K share the layout)
template void tiled_run_microtile<tiled_tile_src0_q4_K>(const tiled_tile_src0_q4_K & src0, const tiled_tile_src1 & src1,
                                                       int i0, int j0, float * buf, int buf_stride);
template void tiled_run_microtile<tiled_tile_src0_q6_K>(const tiled_tile_src0_q6_K & src0, const tiled_tile_src1 & src1,
                                                       int i0, int j0, float * buf, int buf_stride);
template void tiled_run_microtile<tiled_tile_src0_q3_K>(const tiled_tile_src0_q3_K & src0, const tiled_tile_src1 & src1,
                                                       int i0, int j0, float * buf, int buf_stride);
template void tiled_run_microtile<tiled_tile_src0_q2_K>(const tiled_tile_src0_q2_K & src0, const tiled_tile_src1 & src1,
                                                       int i0, int j0, float * buf, int buf_stride);

// Transpose-store the j-major buffer to dst (i contiguous). The buffer's natural layout
// matches the VNNI microtile output (j-major), so per-block accumulation is a cheap
// contiguous add; the one-shot transpose here pays for the i-major dst order. 8x8 blocks
// give contiguous 32B stores to dst (no write amplification).
void tiled_store_window(const float * buf, int n_src0, int n_src1, int buf_stride, float * dst, size_t dst_stride) {
    int ri = 0;
    for (; ri + 8 <= n_src0; ri += 8) {
        int rj = 0;
        for (; rj + 8 <= n_src1; rj += 8) {
#if defined(__AVX2__)
            float r[8][8];
            for (int t = 0; t < 8; t++) {
                _mm256_storeu_ps(&r[t][0], _mm256_loadu_ps(&buf[(ri + t) * buf_stride + rj]));
            }
            for (int u = 0; u < 8; u++) {
                float col[8];
                for (int t = 0; t < 8; t++) col[t] = r[t][u];
                _mm256_storeu_ps(&dst[ri + (size_t)(rj + u) * dst_stride], _mm256_loadu_ps(col));
            }
#else
            for (int u = 0; u < 8; u++) {
                for (int t = 0; t < 8; t++) {
                    dst[(ri + t) + (size_t)(rj + u) * dst_stride] = buf[(ri + t) * buf_stride + (rj + u)];
                }
            }
#endif
        }
        // ragged j tail
        for (; rj < n_src1; rj++) {
            for (int t = 0; t < 8; t++) {
                dst[(ri + t) + (size_t)rj * dst_stride] = buf[(ri + t) * buf_stride + rj];
            }
        }
    }
    // ragged i tail
    for (; ri < n_src0; ri++) {
        for (int j = 0; j < n_src1; j++) {
            dst[ri + (size_t)j * dst_stride] = buf[ri * buf_stride + j];
        }
    }
}