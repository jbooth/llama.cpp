// Tiled matmul microtile kernel: K-quant weights x q8_K activations.
// Templated on the src0 tile config (identical for q4_K/q5_K).
// Per-ISA branches: AVX512-VNNI (dpbusd), AVX2/AVX (maddubs, reference
// vec_dot style), scalar (SSE-only builds).

#include "tiled-kernel.h"
#include "ggml-cpu-impl.h"

#include <string.h>

#if defined(__AVX512VNNI__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

// 12-byte packed scale/min layout of the K-quants, unpack as in the reference kernels
#define TILED_KMASK1 0x3f3f3f3f
#define TILED_KMASK2 0x0f0f0f0f
#define TILED_KMASK3 0x03030303

// 32-byte-unit unpack primitives shared by the src0-format code expansion. All codes are
// bitfields with no overlap, so merging a flag into an already-extracted code uses OR
// (identical to ADD, matches the reference kernels' semantics).
#if defined(__AVX2__)
// packed 4-bit codes -> low nibbles (lo) + high nibbles (hi)
static inline void tiled_unpk_nib4(const uint8_t * src, uint8_t * lo, uint8_t * hi) {
    const __m256i v = _mm256_loadu_si256((const __m256i *) src);
    // mask before the lane shift so bits do not cross byte boundaries
    _mm256_storeu_si256((__m256i *) lo, _mm256_and_si256(v, _mm256_set1_epi8(0x0F)));
    _mm256_storeu_si256((__m256i *) hi, _mm256_srli_epi32(_mm256_and_si256(v, _mm256_set1_epi8(0xF0)), 4));
}
// 2-bit values at bit offset S
template <int S> static inline void tiled_unpk_2bit(const uint8_t * src, uint8_t * dst) {
    _mm256_storeu_si256((__m256i *) dst, _mm256_and_si256(
        _mm256_srli_epi32(_mm256_loadu_si256((const __m256i *) src), S), _mm256_set1_epi8(0x03)));
}
// OR the M-bit value at bit offset S of src into bit offset D of dst
template <int S, int D, int M>
static inline void tiled_unpk_or(uint8_t * dst, const uint8_t * src) {
    const __m256i v = _mm256_slli_epi32(_mm256_and_si256(
        _mm256_srli_epi32(_mm256_loadu_si256((const __m256i *) src), S), _mm256_set1_epi8((uint8_t) M)), D);
    _mm256_storeu_si256((__m256i *) dst, _mm256_or_si256(_mm256_loadu_si256((const __m256i *) dst), v));
}
#else
static inline void tiled_unpk_nib4(const uint8_t * src, uint8_t * lo, uint8_t * hi) {
    for (int l = 0; l < 32; l++) { lo[l] = (uint8_t) (src[l] & 0xF); hi[l] = (uint8_t) (src[l] >> 4); }
}
template <int S>
static inline void tiled_unpk_2bit(const uint8_t * src, uint8_t * dst) {
    for (int l = 0; l < 32; l++) dst[l] = (uint8_t) ((src[l] >> S) & 3);
}
template <int S, int D, int M>
static inline void tiled_unpk_or(uint8_t * dst, const uint8_t * src) {
    for (int l = 0; l < 32; l++) dst[l] = (uint8_t) (dst[l] | (((src[l] >> S) & M) << D));
}
#endif

void tiled_unpack_src0_q4_K(const block_q4_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q4_K * tile) {
    for (int r = 0; r < n_rows; r++) {
        const block_q4_K & x = rows[r * row_stride];

        tile->d[r]    = ggml_fp16_to_fp32(x.d);
        tile->dmin[r] = ggml_fp16_to_fp32(x.dmin);

        uint32_t utmp[4];
        memcpy(utmp, x.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & TILED_KMASK2) | (((utmp[1] >> 6) & TILED_KMASK3) << 4);
        const uint32_t uaux = utmp[1] & TILED_KMASK1;
        utmp[1] = (utmp[2] & TILED_KMASK2) | (((utmp[0] >> 6) & TILED_KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= TILED_KMASK1;

        const uint8_t * sc = (const uint8_t *) &utmp[0];
        const uint8_t * mn = (const uint8_t *) &utmp[2];
        for (int s = 0; s < 8; s++) {
            tile->sc[r * 8 + s] = (int8_t) sc[s];
            tile->mn[r * 8 + s] = (int8_t) mn[s];
        }

        // extract the 4-bit codes (low 4 + high 4), same extraction as the reference kernels
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        tiled_unpk_nib4(x.qs + 0,  q + 0,   q + 32);
        tiled_unpk_nib4(x.qs + 32, q + 64,  q + 96);
        tiled_unpk_nib4(x.qs + 64, q + 128, q + 160);
        tiled_unpk_nib4(x.qs + 96, q + 192, q + 224);
    }
}

void tiled_unpack_src0_q5_K(const block_q5_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q5_K * tile) {
    for (int r = 0; r < n_rows; r++) {
        const block_q5_K & x = rows[r * row_stride];

        tile->d[r]    = ggml_fp16_to_fp32(x.d);
        tile->dmin[r] = ggml_fp16_to_fp32(x.dmin);

        uint32_t utmp[4];
        memcpy(utmp, x.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & TILED_KMASK2) | (((utmp[1] >> 6) & TILED_KMASK3) << 4);
        const uint32_t uaux = utmp[1] & TILED_KMASK1;
        utmp[1] = (utmp[2] & TILED_KMASK2) | (((utmp[0] >> 6) & TILED_KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= TILED_KMASK1;

        const uint8_t * sc = (const uint8_t *) &utmp[0];
        const uint8_t * mn = (const uint8_t *) &utmp[2];
        for (int s = 0; s < 8; s++) {
            tile->sc[r * 8 + s] = (int8_t) sc[s];
            tile->mn[r * 8 + s] = (int8_t) mn[s];
        }

        // extract the 5-bit codes (4 low bits + 1 high bit), same extraction as the reference kernels:
        // 64-element chunk j uses qh bits 2j (low 32) and 2j+1 (high 32); OR adds the 5th bit
        // (no overlap with the 4-bit codes, identical to the reference ADD)
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        tiled_unpk_nib4(x.qs + 0,  q + 0,   q + 32);
        tiled_unpk_nib4(x.qs + 32, q + 64,  q + 96);
        tiled_unpk_nib4(x.qs + 64, q + 128, q + 160);
        tiled_unpk_nib4(x.qs + 96, q + 192, q + 224);
        tiled_unpk_or<0, 4, 1>(q + 0,   x.qh);
        tiled_unpk_or<1, 4, 1>(q + 32,  x.qh);
        tiled_unpk_or<2, 4, 1>(q + 64,  x.qh);
        tiled_unpk_or<3, 4, 1>(q + 96,  x.qh);
        tiled_unpk_or<4, 4, 1>(q + 128, x.qh);
        tiled_unpk_or<5, 4, 1>(q + 160, x.qh);
        tiled_unpk_or<6, 4, 1>(q + 192, x.qh);
        tiled_unpk_or<7, 4, 1>(q + 224, x.qh);
    }
}

void tiled_unpack_src0_q6_K(const block_q6_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q6_K * tile) {
    constexpr int NB = TILED_TILE_K / 16; // 16-wide subblocks, 16 per 256-K
    for (int r = 0; r < n_rows; r++) {
        const block_q6_K & x = rows[r * row_stride];
        tile->d[r] = ggml_fp16_to_fp32(x.d);

        // 6-bit code = 4 low bits (ql) | 2 high bits (qh); see ggml_vec_dot_q6_K_q8_K_generic
        // per half the lanes are [ql lo(0:32)] [ql lo(32:64)] [ql hi(0:32)] [ql hi(32:64)]
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        for (int half = 0; half < 2; half++) {
            uint8_t * out = q + 128 * half;
            tiled_unpk_nib4(x.ql + 64 * half + 0,  out + 0,  out + 64);
            tiled_unpk_nib4(x.ql + 64 * half + 32, out + 32, out + 96);
            tiled_unpk_or<0, 4, 3>(out + 0,  x.qh + 32 * half);
            tiled_unpk_or<2, 4, 3>(out + 32, x.qh + 32 * half);
            tiled_unpk_or<4, 4, 3>(out + 64, x.qh + 32 * half);
            tiled_unpk_or<6, 4, 3>(out + 96, x.qh + 32 * half);
        }
// scale is a plain int8 per 16-element subblock (16 per 256-K)
        for (int s = 0; s < NB; s++)
            tile->sc[r * NB + s] = (int8_t) x.scales[s];
    }
}

void tiled_unpack_src0_q3_K(const block_q3_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q3_K * tile) {
    constexpr int NB = TILED_TILE_K / 16;
    for (int r = 0; r < n_rows; r++) {
        const block_q3_K & x = rows[r * row_stride];
        tile->d[r] = ggml_fp16_to_fp32(x.d);

        // 3-bit code = 2 low bits (qs) | (1 high bit from hmask << 2)
        // element e (0..255): half=e>>7, el=e&127, group=el>>5, l=el&31
        //   low2 = (qs[half*32 + l] >> 2*group) & 3
        //   high = (hmask[l] >> (half*4 + group)) & 1
        // see ggml_vec_dot_q3_K_q8_K_generic
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        const uint8_t * s0 = x.qs;
        const uint8_t * s1 = x.qs + 32;
        uint8_t * o0 = q;
        uint8_t * o1 = q + 128;
        tiled_unpk_2bit<0>(s0, o0);      tiled_unpk_or<0, 2, 1>(o0,      x.hmask);
        tiled_unpk_2bit<2>(s0, o0 + 32); tiled_unpk_or<1, 2, 1>(o0 + 32, x.hmask);
        tiled_unpk_2bit<4>(s0, o0 + 64); tiled_unpk_or<2, 2, 1>(o0 + 64, x.hmask);
        tiled_unpk_2bit<6>(s0, o0 + 96); tiled_unpk_or<3, 2, 1>(o0 + 96, x.hmask);
        tiled_unpk_2bit<0>(s1, o1);      tiled_unpk_or<4, 2, 1>(o1,      x.hmask);
        tiled_unpk_2bit<2>(s1, o1 + 32); tiled_unpk_or<5, 2, 1>(o1 + 32, x.hmask);
        tiled_unpk_2bit<4>(s1, o1 + 64); tiled_unpk_or<6, 2, 1>(o1 + 64, x.hmask);
        tiled_unpk_2bit<6>(s1, o1 + 96); tiled_unpk_or<7, 2, 1>(o1 + 96, x.hmask);
        // 6-bit scale decode (same kmask trick as the reference), stored as (sc - 32)
        uint32_t auxs[4];
        memcpy(auxs, x.scales, 12);
        const uint32_t kmask1 = 0x03030303;
        const uint32_t kmask2 = 0x0f0f0f0f;
        const uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t * scales = (const int8_t *) &auxs[0];
        for (int s = 0; s < NB; s++)
            tile->sc[r * NB + s] = (int8_t) (scales[s] - 32);
    }
}

void tiled_unpack_src0_q2_K(const block_q2_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q2_K * tile) {
    constexpr int NB = TILED_TILE_K / 16;
    for (int r = 0; r < n_rows; r++) {
        const block_q2_K & x = rows[r * row_stride];
        tile->d[r]    = ggml_fp16_to_fp32(x.d);
        tile->dmin[r] = ggml_fp16_to_fp32(x.dmin);

        // 2-bit code: element e -> half=e>>7, el=e&127
        //   byte = half*32 + (el & 31), shift = 2*(el >> 5)
        // see ggml_vec_dot_q2_K_q8_K_generic
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        for (int half = 0; half < 2; half++) {
            const uint8_t * s = x.qs + 32 * half;
            uint8_t * out = q + 128 * half;
            tiled_unpk_2bit<0>(s, out + 0);
            tiled_unpk_2bit<2>(s, out + 32);
            tiled_unpk_2bit<4>(s, out + 64);
            tiled_unpk_2bit<6>(s, out + 96);
        }
        // scale/min packed in one byte per 16-element subblock: low 4 bits = scale, high 4 = min
        for (int s = 0; s < NB; s++) {
            tile->sc[r * NB + s] = (int8_t) (x.scales[s] & 0xF);
            tile->mn[r * NB + s] = (int8_t) (x.scales[s] >> 4);
        }
    }
}

void tiled_interleave_src1_q8_K(const block_q8_K * rows, int64_t row_stride,
                                int64_t r_start, int64_t r_end,
                                int64_t n_k, int64_t n_rows, int64_t n_rows_pad, int8_t * qv_glob) {
    // n_k is the region width in elements (already slab-padded by the caller);
    // the last slab may be partial and its excess groups read the row's tail
    const int64_t n_slabs = n_k / TILED_TILE_K;
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
    // the per-call VNNI unpack shape (16-lane 4B gather + 64B store per
    // k-group x 16-row block), run once over the whole tensor
    static_assert(sizeof(block_q8_K) == 292, "block_q8_K size changed, fix the gather indices below");
    static_assert(offsetof(block_q8_K, qs) == 4, "block_q8_K qs offset changed, fix the gather indices below");
    {
        // row_off[r] = int32 index (scale 4) of rows[r].qs[0] (d is 4B, qs at +4);
        // row_stride * 73 stays in int32 for any realistic row count
        const int32_t s73 = (int32_t) row_stride * 73;
        int32_t row_off[TILED_MICRO];
        for (int r = 0; r < TILED_MICRO; r++) row_off[r] = r * s73 + 1;
        const __m512i idx_row = _mm512_loadu_si512((const __m512i *) row_off);
        const int32_t * base = (const int32_t *) rows;
        for (int64_t r0 = r_start; r0 < r_end; r0 += TILED_MICRO) {
            // masked lanes (past n_rows) gather into the zero register, so the
            // 16-row pad tail of the region is zeroed here, no separate memset
            const int64_t n = (n_rows > r0) ? n_rows - r0 : 0;
            const __mmask16 k = (n >= TILED_MICRO) ? 0xffff : ((__mmask16) ((1u << n) - 1));
            const int64_t rl = r0 - r_start; // rows points at global row r_start
            for (int64_t s = 0; s < n_slabs; s++) {
                // block (row rl, slab s) is rl*row_stride + s blocks from rows
                const __m512i base_idx = _mm512_set1_epi32((int32_t) ((rl * row_stride + s) * 73));
                int8_t * out = qv_glob + s * TILED_TILE_K * n_rows_pad + r0 * 4;
                for (int g = 0; g < 64; g++) {
                    const __m512i idx = _mm512_add_epi32(_mm512_add_epi32(idx_row, base_idx), _mm512_set1_epi32(g));
                    const __m512i v = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), k, idx, base, 4);
                    _mm512_storeu_si512((void *) (out + g * n_rows_pad * 4), v);
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
                int8_t * out = qv_glob + (s * TILED_TILE_K) * n_rows_pad + r * 4;
                if (r < n_rows) {
                    const block_q8_K & x = rows[(r - r_start) * row_stride + s];
                    for (int g = 0; g < 64; g++) {
                        memcpy(out + g * n_rows_pad * 4, x.qs + g * 4, 4);
                    }
                } else {
                    for (int g = 0; g < 64; g++) {
                        memset(out + g * n_rows_pad * 4, 0, 4);
                    }
                }
            }
        }
    }
}

void tiled_unpack_src1_q8_K(const block_q8_K * rows, int64_t row_stride, int n_rows, tiled_tile_src1 * tile,
                            const int8_t * qv_glob, int64_t n_rows_pad, int64_t r_glob, int64_t kblk) {
    const int n_padded = (n_rows + TILED_MICRO - 1) & ~(TILED_MICRO - 1);
    const int n_qv = TILED_TILE_K / 4;
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
    // The VNNI body reads only qv, never q, so on this build we build qv alone.
    if (qv_glob) {
        // the tensor was interleaved once (tiled_interleave_src1_q8_K): the tile
        // column is 4 bytes per row, contiguous; rows past n_rows read the
        // zeroed pad tail of the region
        for (int g = 0; g < n_qv; g++) {
            memcpy(&tile->qv[g * TILED_TILE_ROWS * 4],
                   qv_glob + ((int64_t) (kblk * n_qv + g) * n_rows_pad + r_glob) * 4, (size_t) n_padded * 4);
        }
    } else {
        const int32_t s73 = (int32_t) row_stride * 73;
        int32_t row_off[TILED_MICRO];
        for (int r = 0; r < TILED_MICRO; r++) row_off[r] = r * s73 + 1;
        const __m512i idx_row = _mm512_loadu_si512((const __m512i *) row_off);
        const int32_t * base = (const int32_t *) rows;
        for (int r0 = 0; r0 < n_padded; r0 += TILED_MICRO) {
            const int n = (n_rows > r0) ? n_rows - r0 : 0;
            const __mmask16 k = (n >= TILED_MICRO) ? 0xffff : ((__mmask16) ((1u << n) - 1));
            const __m512i base_idx = _mm512_set1_epi32(r0 * s73);
            for (int g = 0; g < n_qv; g++) {
                const __m512i idx = _mm512_add_epi32(_mm512_add_epi32(idx_row, base_idx), _mm512_set1_epi32(g));
                const __m512i v = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), k, idx, base, 4);
                _mm512_storeu_si512((void *) &tile->qv[g * TILED_TILE_ROWS * 4 + r0 * 4], v);
            }
        }
    }
#endif
    // d and bsums: natural layout, every build
    for (int r = 0; r < n_padded; r++) {
        if (r < n_rows) {
            const block_q8_K & x = rows[r * row_stride];
            for (int s = 0; s < TILED_TILE_K / 16; s++) {
                tile->bsums[s * TILED_TILE_ROWS + r] = x.bsums[s];
            }
            tile->d[r] = x.d;
        } else {
            for (int s = 0; s < TILED_TILE_K / 16; s++) {
                tile->bsums[s * TILED_TILE_ROWS + r] = 0;
            }
            tile->d[r] = 0.0f;
        }
    }
#if !defined(__AVX512VNNI__) || !defined(__AVX512VL__) || !defined(__AVX512DQ__)
    // non-VNNI build: the scalar/AVX2 bodies read q ([row][k]); qv is not built
    for (int r = 0; r < n_padded; r++) {
        if (r < n_rows) {
            const block_q8_K & x = rows[r * row_stride];
            memcpy(&tile->q[r * TILED_TILE_K], x.qs, TILED_TILE_K);
            // VNNI interleaved copy: [k/4][row][4], a 32B-aligned row-tile load per k/4 group
            const int8_t * qs = x.qs;
            for (int g = 0; g < n_qv; g++) {
                memcpy(&tile->qv[g * TILED_TILE_ROWS * 4 + r * 4], qs + g * 4, 4);
            }
        } else {
            memset(&tile->q[r * TILED_TILE_K], 0, TILED_TILE_K);
            for (int g = 0; g < n_qv; g++) {
                memset(&tile->qv[g * TILED_TILE_ROWS * 4 + r * 4], 0, 4);
            }
        }
    }
#endif
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
            int32_t bs = 0;
            for (int u = 0; u < NS; u++) {
                bs += src1.bsums[(s * NS + u) * TILED_TILE_ROWS + br];
            }

            for (int i = 0; i < TILED_MICRO; i++) {
                const int ar = i0 + i;
                const uint8_t * q0 = &src0.q[ar * TILED_TILE_K + s * SUBBLK];

                int32_t raw = 0;
                for (int e = 0; e < SUBBLK; e++) {
                    raw += (int32_t) q0[e] * (int32_t) q1[e];
                }

                // BIAS: subtract BIAS*bs (src1's per-subblock code sum) from the
                // exact int raw (q3_K=4, q6_K=32; 0 otherwise, vanishes via constexpr)
                int32_t corr = raw;
                if constexpr (BIAS != 0)
                    corr -= BIAS * bs;
                const int32_t sc_raw = (int32_t) src0.sc[ar * NB + s] * corr;
                // d1 is NOT applied here: it is constant over the s-loop, so
                // acc holds the d1-un-scaled sum and the store below applies
                // d1 once per element (same factoring as the VNNI kernel)
                if constexpr (HAS_MIN) {
                    const int32_t mn_bs = (int32_t) src0.mn[ar * NB + s] * bs;
                    acc[i][j] += (float) src0.d[ar] * (float) sc_raw
                              - (float) src0.dmin[ar] * (float) mn_bs;
                } else {
                    acc[i][j] += (float) src0.d[ar] * (float) sc_raw;
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

// =====================================================================
// VNNI microtile kernel
// =====================================================================
//
// AVX-512 basics used by this function
// ------------------------------------
// src0 512-bit register (the __m512 or __m512i C type, "ZMM" register in
// Intel's naming) is 64 bytes. The C type is only a hint to the
// compiler; the same 64 physical bytes are read differently depending
// on which intrinsics touch them:
//   - __m512  as 16 x float32 lanes
//   - __m512i as 16 x int32 lanes
//   - 64 x int8 only inside the dpbusd dot instruction
// The machine's register file holds 32 of these registers. This kernel
// is register-pressure-driven: the int accumulators (acc16[] + the split-pass
// S1acc/S2acc[]) + a handful of live temps must all stay resident in those 32
// registers. If any accumulator spills to the
// stack, every dpbusd turns into a read-modify-write through memory
// and the kernel gets SLOWER than the scalar one (that is exactly what
// happened to the 16x16 microtile in the v1 attempt, see plan section
// 7.2: an R x C microtile needs R*C/16 int + R*C/16 float ZMM, which
// for 16x16 is all 32 registers and nothing left for temps).
//
// VNNI ("Vector Neural Network Instructions") is the AVX-512 extension
// we use here; the one member we need is vpdpbusd,
// the _mm512_dpbusd_epi32() intrinsic. Its full semantics:
//
//     for i = 0..15 (one per 32-bit lane):
//         dst[i] += (uint8)a[4i+0]* (int8)b[4i+0]
//                 + (uint8)a[4i+1]* (int8)b[4i+1]
//                 + (uint8)a[4i+2]* (int8)b[4i+2]
//                 + (uint8)a[4i+3]* (int8)b[4i+3]

// In words: each destination lane takes the 4 bytes of that lane from
// each source, multiplies them unsigned x signed, sums the 4 products,
// and adds them into the lane. One instruction therefore does 64
// byte-multiplications and 16 int32 accumulations. The first argument
// is the accumulator (the "add form" of the instruction). The
// unsigned/signed split is a hardware fact: src0 must hold the unsigned
// weight codes, src1 the signed q8 (plan section 4).
//
// Why the 16 lanes mean 16 src1-columns instead of 16 k-elements: for the
// 16 lanes to be 16 different output pairs (outer-product structure),
// every lane of the src0 register must hold the SAME 4 src0-codes (one
// src0-row's codes for this 4-k group, broadcast), so that lane j pairs
// them with a different src1 column's 4 codes. The src1 register therefore
// holds 16 distinct columns. That is why src1 is pre-interleaved at
// unpack time into the qv layout [k/4][row][4]: for a fixed 4-k group
// kg and a run of 16 src1-columns, the needed bytes are exactly one
// contiguous 64B chunk, so the src1-operand is a single load that is
// reused across all 8 src0-rows, and the src0-operand is a cheap scalar
// broadcast.
//
// Math split (plan section 7.1). The subblock contribution to
// dst[i][j] is
//     src1.d[j] * ( src0.d[i]*sc[i][s]*raw - src0.dmin[i]*mn[i][s]*bs[j] )
// where raw is the exact integer dot of the unsigned src0 codes with the
// signed src1 codes over the SUBBLK elements, and bs[j] is the sum of the
// src1 codes over the same subblock (needed for the min correction). The
// correction is int split-pass: per subblock, S1 += sc*raw and
// S2 += mn*bs accumulate in int32 on the INT pipe (mullo + padd), and
// those int accumulators live across the whole 256-K block. The int->
// float convert and the per-row d0/dmin/d1 application happen once per
// row at the epilogue. The correction was originally float (cvt +
// fmadd/fnmadd on the FMA pipe, which also runs dpbusd on Zen 5); the
// int version measured 3-12% faster and the float one was deleted
// (plan v2.5, tiled-bench-results.md).
//
// Microtile shape: 16x16 (matching the driver's microtile window). The band
// pass below computes NA x 16 (NA = 8 for all formats: acc16 + S1acc = 16
// zmm + ~4 temps; the S2 term is hoisted out of the s loop, plan 12.5, so
// HAS_MIN no longer needs its own S2acc zmm), and a short wrapper runs 2
// band passes to cover the 16 src0-rows. A full 16-row band in one pass
// needs 32 zmm and was rejected for the register-count reason above.
#define TILED_VNNI_BAND 8
#define TILED_VNNI_INT_BAND 4 // pre-12.5 band for HAS_MIN (dead, kept for the
                              // unmeasured _optimized variant below)

// Alternate kernel, was supposed to improve performance but did not.
template <typename T>
static void tiled_run_micro_vnni_int_optimized(const T & src0, const tiled_tile_src1 & src1,
                                                int i0, int j0, float * buf, int buf_stride) {
    constexpr int NB = T::NB;
    constexpr int SUBBLK = TILED_TILE_K / NB;
    constexpr int NS = SUBBLK / 16;
    constexpr int NG = SUBBLK / 4;
    constexpr bool HAS_MIN = T::HAS_MIN_V;
    constexpr int BIAS = T::BIAS_V;
    constexpr int NA = HAS_MIN ? TILED_VNNI_INT_BAND : TILED_VNNI_BAND;

    const __m512 d1_vec = _mm512_loadu_ps(&src1.d[j0]);

    __m512i S1acc[NA];
    __m512i S2acc[NA];

    for (int t = 0; t < NA; t++) {
        S1acc[t] = _mm512_setzero_si512();
        if constexpr (HAS_MIN) S2acc[t] = _mm512_setzero_si512();
    }

    for (int s = 0; s < NB; s++) {
        // --- 1. Vectorized src1-side block sum calculation ---
        __m512i bs32 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *) &src1.bsums[s * TILED_TILE_ROWS * NS + j0]));
        for (int u = 1; u < NS; u++) {
            const __m256i bsum_16 = _mm256_loadu_si256((const __m256i *) &src1.bsums[(s * NS + u) * TILED_TILE_ROWS + j0]);
            bs32 = _mm512_add_epi32(bs32, _mm512_cvtepi16_epi32(bsum_16));
        }

        __m512i acc16[NA];
        for (int t = 0; t < NA; t++) acc16[t] = _mm512_setzero_si512();

        // --- 2. Main Compute Loop: Pre-fetch & Pipeline Interleaving ---
        for (int g = 0; g < NG; g++) {
            const int kg = s * NG + g;
            
            // Single 512-bit load for src1 matrix activations
            const __m512i qv = _mm512_loadu_si512((const __m512i *) &src1.qv[kg * TILED_TILE_ROWS * 4 + j0 * 4]);

            // Unrolled execution to keep FMA/VNNI pipelines 100% saturated
            #pragma unroll
            for (int t = 0; t < NA; t++) {
                // Vector-friendly broadcast load
                const int32_t a_val = *(const int32_t *) &src0.q[(i0 + t) * TILED_TILE_K + kg * 4];
                const __m512i u4b = _mm512_set1_epi32(a_val);
                
                // Back-to-back dpbusd with zero register dependency between t steps
                acc16[t] = _mm512_dpbusd_epi32(acc16[t], u4b, qv);
            }
        }

        // --- 3. Subblock Correction Phase ---
        __m512i bias32 = _mm512_setzero_si512();
        if constexpr (BIAS != 0) {
            bias32 = _mm512_mullo_epi32(bs32, _mm512_set1_epi32(BIAS));
        }

        #pragma unroll
        for (int t = 0; t < NA; t++) {
            const int ar = i0 + t;
            __m512i rawi = acc16[t];
            if constexpr (BIAS != 0) {
                rawi = _mm512_sub_epi32(rawi, bias32);
            }

            const __m512i sc_vec = _mm512_set1_epi32((int) src0.sc[ar * NB + s]);
            S1acc[t] = _mm512_add_epi32(S1acc[t], _mm512_mullo_epi32(rawi, sc_vec));

            if constexpr (HAS_MIN) {
                const __m512i mn_vec = _mm512_set1_epi32((int) src0.mn[ar * NB + s]);
                S2acc[t] = _mm512_add_epi32(S2acc[t], _mm512_mullo_epi32(bs32, mn_vec));
            }
        }
    }

    // --- 4. Epilogue: Single FMA Store ---
    #pragma unroll
    for (int t = 0; t < NA; t++) {
        const int ar = i0 + t;
        const __m512 f1 = _mm512_cvtepi32_ps(S1acc[t]);
        __m512 result = _mm512_mul_ps(f1, _mm512_set1_ps(src0.d[ar]));

        if constexpr (HAS_MIN) {
            const __m512 f2 = _mm512_cvtepi32_ps(S2acc[t]);
            result = _mm512_fnmadd_ps(_mm512_set1_ps(src0.dmin[ar]), f2, result);
        }

        float * p = &buf[(i0 + t) * buf_stride + j0];
        const __m512 cur = _mm512_loadu_ps(p);
        _mm512_storeu_ps(p, _mm512_fmadd_ps(result, d1_vec, cur));
    }
}

// =====================================================================
// VNNI microtile kernel, int split-pass correction
// =====================================================================
//
// The per-subblock correction is done in int32 instead of float. sc and
// mn are int8 in the tile, so sc*raw and mn*bs are exact int32 products.
// The per-row floats d0/dmin are applied once at the epilogue instead
// of once per (row, subblock).
//
// The correction runs on the INT pipe (3 ports, no contention with
// dpbusd, which is on the FMA pipe on Zen 5) instead of the FMA pipe.
// Per-subblock cost is mullo+padd (2 INT pipe ops, 4 with HAS_MIN).
// Measured 3-12% faster than the float version on the large-K shape;
// that version and the benchmark switch are deleted (plan v2.5,
// tiled-bench-results.md).
//
// Register pressure: S1acc + acc16 = 16 zmm (no-min, 8-row band);
// S1acc + S2acc + acc16 = 12 zmm (HAS_MIN, 4-row band). Fits the 32-
// ZMM file without spills (a wider band spills and runs slower).
//
// Math: S1 += sc_s*raw_s, S2 += mn_s*bs_s per subblock; the row result
// is d0*S1 - dmin*S2 (f32, exact in range), applied once per row.

 template <typename T>
static void tiled_run_micro_vnni_int(const T & src0, const tiled_tile_src1 & src1,
                                     int i0, int j0, float * buf, int buf_stride) {
    constexpr int NB = T::NB;
    constexpr int SUBBLK = TILED_TILE_K / NB;
    constexpr int NS = SUBBLK / 16;
    constexpr int NG = SUBBLK / 4;
    constexpr bool HAS_MIN = T::HAS_MIN_V;
    constexpr int BIAS = T::BIAS_V;

    constexpr int NA = TILED_VNNI_BAND; // scratch 12.5: S2 hoisted, HAS_MIN runs NA=8

    const __m512 d1_vec = _mm512_loadu_ps(&src1.d[j0]);

    __m512i S1acc[NA];
    for (int t = 0; t < NA; t++) S1acc[t] = _mm512_setzero_si512();
    __m512i S2acc[NA];
    if constexpr (HAS_MIN)
        for (int t = 0; t < NA; t++) S2acc[t] = _mm512_setzero_si512();

    for (int s = 0; s < NB; s++) {
        __m512i bs32 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *) &src1.bsums[s * TILED_TILE_ROWS * NS + j0]));
        for (int u = 1; u < NS; u++) {
            bs32 = _mm512_add_epi32(bs32, _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *) &src1.bsums[(s * NS + u) * TILED_TILE_ROWS + j0])));
        }

        __m512i bias32 = _mm512_setzero_si512();
        if constexpr (BIAS != 0)
            bias32 = _mm512_mullo_epi32(bs32, _mm512_set1_epi32(BIAS));

        __m512i acc16[NA];
        for (int t = 0; t < NA; t++) acc16[t] = _mm512_setzero_si512();

        // scratch 12.5: force the g-loop open (64-dpbusd body would otherwise stay a loop with acc rotation)
        #pragma GCC unroll 8
        for (int g = 0; g < NG; g++) {
            const int kg = s * NG + g;
            const __m512i qv = _mm512_loadu_si512((const __m512i *) &src1.qv[kg * TILED_TILE_ROWS * 4 + j0 * 4]);
            for (int t = 0; t < NA; t++) {
                const uint32_t u4 = *(const uint32_t *) &src0.q[(i0 + t) * TILED_TILE_K + kg * 4];
                const __m512i u4b = _mm512_set1_epi32((int) u4);
                acc16[t] = _mm512_dpbusd_epi32(acc16[t], u4b, qv);
            }
        }

        // int correction: S1 += sc*(raw-BIAS*bs), S2 += mn*bs
        for (int t = 0; t < NA; t++) {
            const int ar = i0 + t;
            __m512i rawi = acc16[t];
            if constexpr (BIAS != 0)
                rawi = _mm512_sub_epi32(rawi, bias32);
            S1acc[t] = _mm512_add_epi32(S1acc[t],
                _mm512_mullo_epi32(rawi, _mm512_set1_epi32((int) src0.sc[ar * NB + s])));
        }
    }

    // scratch 12.5: S2 += mn*bs, no dpbusd dependency, kept out of the band pass
    if constexpr (HAS_MIN) {
        for (int s = 0; s < NB; s++) {
            __m512i bs32 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *) &src1.bsums[s * TILED_TILE_ROWS * NS + j0]));
            for (int u = 1; u < NS; u++) {
                bs32 = _mm512_add_epi32(bs32, _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *) &src1.bsums[(s * NS + u) * TILED_TILE_ROWS + j0])));
            }
            for (int t = 0; t < NA; t++) {
                S2acc[t] = _mm512_add_epi32(S2acc[t],
                    _mm512_mullo_epi32(bs32, _mm512_set1_epi32((int) src0.mn[(i0 + t) * NB + s])));
            }
        }
    }

    // epilogue: int->float, apply per-row scales, store to buf
    for (int t = 0; t < NA; t++) {
        const int ar = i0 + t;
        __m512 f1 = _mm512_cvtepi32_ps(S1acc[t]);
        __m512 result = _mm512_mul_ps(f1, _mm512_set1_ps(src0.d[ar]));
        if constexpr (HAS_MIN) {
            __m512 f2 = _mm512_cvtepi32_ps(S2acc[t]);
            result = _mm512_fnmadd_ps(_mm512_set1_ps(src0.dmin[ar]), f2, result);
        }
        float * p = &buf[(i0 + t) * buf_stride + j0];
        _mm512_storeu_ps(p, _mm512_add_ps(_mm512_loadu_ps(p), _mm512_mul_ps(result, d1_vec)));
    }
}

// 16 src0-row microtile as NA-row band passes: 2 passes of 8x16 (plan 12.5)
 template <typename T>
static void tiled_run_microtile_vnni_int(const T & src0, const tiled_tile_src1 & src1,
                                         int i0, int j0, float * buf, int buf_stride) {
    constexpr int NA = TILED_VNNI_BAND; // scratch 12.5
    for (int i = 0; i < TILED_MICRO; i += NA) {
        tiled_run_micro_vnni_int(src0, src1, i0 + i, j0, buf, buf_stride);
    }
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
// accumulator per column, live across the subblock loop; GROUP = 4
// fits the 16 YMM budget (the accs + the a/b/sc/p16 operands). The src0
// subblock codes are loaded once per (i, s, group) and reused over the
// group: 1/4 of the src0-side load traffic of a pure per-pair loop. The
// 16x16 microtile only fixes the buf layout.
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
// accumulators hold sc*raw over the subblocks, so the -> scalar reduce
// happens ONCE per pair (in the group epilogue). The per-subblock BIAS
// and mn corrections run as 128-bit i32 vectors, one lane per src1 column,
// with bs from the per-16 bsum table (v2.8: these were scalar per
// column and dominated the SUBBLK=16 body). S1 = hsum(acc) - BIAS*sum
// (sc*bs), S2 = sum(mn*bs); the epilogue is scalar: buf += d1 * (d0*S1
// - dmin*S2). No FMA is used (the epilogue is scalar), so the AVX2 tier
// alone is sufficient.

 template <typename T>
static void tiled_run_microtile_avx2(const T & src0, const tiled_tile_src1 & src1,
                                     int i0, int j0, float * buf, int buf_stride) {
    constexpr int NB = T::NB;
    constexpr int SUBBLK = TILED_TILE_K / NB;
    constexpr bool HAS_MIN = T::HAS_MIN_V;
    constexpr int BIAS = T::BIAS_V;
    constexpr int GROUP = 4; // src1 columns per group: one acc32 per column

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
        const int8_t * sc_row = &src0.sc[ar * NB];
        const int8_t * mn_row = &src0.mn[ar * NB];

        for (int g = 0; g < TILED_MICRO; g += GROUP) {
            const int8_t * q1g[GROUP];
            __m128i corr = _mm_setzero_si128(); // -BIAS * sum_s sc_s * bs_s, 1 lane per col (vanishes via constexpr)
            __m128i S2 = _mm_setzero_si128();   // sum_s mn_s * bs_s
            if constexpr (SUBBLK == 32) {
                __m256i acc[GROUP];
                for (int t = 0; t < GROUP; t++) {
                    q1g[t] = &src1.q[(j0 + g + t) * TILED_TILE_K];
                    acc[t] = _mm256_setzero_si256();
                }
                // one 32B a load and one 256-bit maddubs per (s, column)
                for (int s = 0; s < NB; s++) {
                    const __m256i q0_32 = _mm256_loadu_si256((const __m256i *) &q0[s * SUBBLK]);
                    const __m256i sc16 = _mm256_set1_epi16(sc_row[s]);
                    const int16_t * bs0 = &src1.bsums[s * 2 * TILED_TILE_ROWS + j0 + g];
                    const int16_t * bs1 = &src1.bsums[(s * 2 + 1) * TILED_TILE_ROWS + j0 + g];
                    // per-col bs as a 4-lane i32 vector: one 8-byte load per bsum row (NS=2)
                    const __m128i bsv = _mm_add_epi32(
                        _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *) bs0)),
                        _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *) bs1)));
                    for (int t = 0; t < GROUP; t++) {
                        acc[t] = _mm256_add_epi32(acc[t],
                            _mm256_madd_epi16(sc16, _mm256_maddubs_epi16(
                                q0_32, _mm256_loadu_si256((const __m256i *) &q1g[t][s * SUBBLK]))));
                    }
                    if constexpr (BIAS != 0)
                        corr = _mm_sub_epi32(corr, _mm_mullo_epi32(bsv, _mm_set1_epi32(BIAS * sc_row[s])));
                    if constexpr (HAS_MIN)
                        S2 = _mm_add_epi32(S2, _mm_mullo_epi32(bsv, _mm_set1_epi32(mn_row[s])));
                }
                int32_t corr_s[GROUP] = { 0, 0, 0, 0 };
                int32_t S2_s[GROUP] = { 0, 0, 0, 0 };
                if constexpr (BIAS != 0)
                    _mm_storeu_si128((__m128i *) corr_s, corr);
                if constexpr (HAS_MIN)
                    _mm_storeu_si128((__m128i *) S2_s, S2);
                for (int t = 0; t < GROUP; t++) {
                    // 8 i32 lanes -> scalar (the per-pair dot, pre-correction)
                    const __m128i lo = _mm256_castsi256_si128(acc[t]);
                    const int32_t S1 = hsum4(_mm_add_epi32(lo, _mm256_extracti128_si256(acc[t], 1))) + corr_s[t];
                    float res = d0 * (float) S1;
                    if constexpr (HAS_MIN)
                        res -= dmin0 * (float) S2_s[t];
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
                    const __m256i scv = _mm256_set_m128i(_mm_set1_epi16(sc_row[sp + 1]), _mm_set1_epi16(sc_row[sp]));
                    const int16_t * bs0 = &src1.bsums[sp * TILED_TILE_ROWS + j0 + g];
                    const int16_t * bs1 = &src1.bsums[(sp + 1) * TILED_TILE_ROWS + j0 + g];
                    const __m128i bs0v = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *) bs0));
                    const __m128i bs1v = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *) bs1));
                    for (int t = 0; t < GROUP; t++) {
                        acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(
                            scv, _mm256_maddubs_epi16(
                                q0_32, _mm256_loadu_si256((const __m256i *) &q1g[t][sp * SUBBLK]))));
                    }
                    // per-subblock scales differ, so corr/S2 keep the bs0/bs1 split
                    if constexpr (BIAS != 0)
                        corr = _mm_sub_epi32(corr, _mm_add_epi32(
                            _mm_mullo_epi32(bs0v, _mm_set1_epi32(BIAS * sc_row[sp])),
                            _mm_mullo_epi32(bs1v, _mm_set1_epi32(BIAS * sc_row[sp + 1]))));
                    if constexpr (HAS_MIN)
                        S2 = _mm_add_epi32(S2, _mm_add_epi32(
                            _mm_mullo_epi32(bs0v, _mm_set1_epi32(mn_row[sp])),
                            _mm_mullo_epi32(bs1v, _mm_set1_epi32(mn_row[sp + 1]))));
                }
                int32_t corr_s[GROUP] = { 0, 0, 0, 0 };
                int32_t S2_s[GROUP] = { 0, 0, 0, 0 };
                if constexpr (BIAS != 0)
                    _mm_storeu_si128((__m128i *) corr_s, corr);
                if constexpr (HAS_MIN)
                    _mm_storeu_si128((__m128i *) S2_s, S2);
                for (int t = 0; t < GROUP; t++) {
                    // 8 i32 lanes -> scalar (the per-pair dot, pre-correction)
                    const __m128i lo = _mm256_castsi256_si128(acc[t]);
                    const int32_t S1 = hsum4(_mm_add_epi32(lo, _mm256_extracti128_si256(acc[t], 1))) + corr_s[t];
                    float res = d0 * (float) S1;
                    if constexpr (HAS_MIN)
                        res -= dmin0 * (float) S2_s[t];
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
// loop, src0 codes hoisted per group); the register budget is looser
// here, so GROUP = 4 is kept for code uniformity.

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
        const int8_t * sc_row = &src0.sc[ar * NB];
        const int8_t * mn_row = &src0.mn[ar * NB];

        for (int g = 0; g < TILED_MICRO; g += GROUP) {
            const int8_t * q1g[GROUP];
            __m128i acc[GROUP];
            __m128i corr = _mm_setzero_si128(); // -BIAS * sum_s sc_s * bs_s, 1 lane per col (vanishes via constexpr)
            __m128i S2 = _mm_setzero_si128();   // sum_s mn_s * bs_s
            for (int t = 0; t < GROUP; t++) {
                q1g[t] = &src1.q[(j0 + g + t) * TILED_TILE_K];
                acc[t] = _mm_setzero_si128();
            }

            for (int s = 0; s < NB; s++) {
                const __m128i sc16 = _mm_set1_epi16(sc_row[s]);
                __m128i bsv = _mm_setzero_si128(); // per-col bs as a 4-lane i32 vector
                for (int u = 0; u < NS; u++) {
                    const __m128i a16 = _mm_loadu_si128((const __m128i *) &q0[s * SUBBLK + u * 16]);
                    bsv = _mm_add_epi32(bsv, _mm_cvtepi16_epi32(_mm_loadl_epi64(
                        (const __m128i *) &src1.bsums[(s * NS + u) * TILED_TILE_ROWS + j0 + g])));
                    for (int t = 0; t < GROUP; t++) {
                        acc[t] = _mm_add_epi32(acc[t], _mm_madd_epi16(sc16,
                            _mm_maddubs_epi16(a16,
                                _mm_loadu_si128((const __m128i *) &q1g[t][s * SUBBLK + u * 16]))));
                    }
                }
                if constexpr (BIAS != 0)
                    corr = _mm_sub_epi32(corr, _mm_mullo_epi32(bsv, _mm_set1_epi32(BIAS * sc_row[s])));
                if constexpr (HAS_MIN)
                    S2 = _mm_add_epi32(S2, _mm_mullo_epi32(bsv, _mm_set1_epi32(mn_row[s])));
            }

            int32_t corr_s[GROUP] = { 0, 0, 0, 0 };
            int32_t S2_s[GROUP] = { 0, 0, 0, 0 };
            if constexpr (BIAS != 0)
                _mm_storeu_si128((__m128i *) corr_s, corr);
            if constexpr (HAS_MIN)
                _mm_storeu_si128((__m128i *) S2_s, S2);

            for (int t = 0; t < GROUP; t++) {
                // 4 i32 lanes -> scalar (the per-pair dot, pre-correction)
                __m128i v = _mm_shuffle_epi32(acc[t], _MM_SHUFFLE(2, 3, 0, 1));
                acc[t] = _mm_add_epi32(acc[t], v);
                v = _mm_shuffle_epi32(acc[t], _MM_SHUFFLE(1, 0, 3, 2));
                acc[t] = _mm_add_epi32(acc[t], v);
                const int32_t S1 = _mm_cvtsi128_si32(acc[t]) + corr_s[t];

                float res = d0 * (float) S1;
                if constexpr (HAS_MIN)
                    res -= dmin0 * (float) S2_s[t];
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
void tiled_store_window(const float * buf, int n_src0, int n_src1, int buf_stride, float * c, size_t ldc) {
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
                _mm256_storeu_ps(&c[ri + (size_t)(rj + u) * ldc], _mm256_loadu_ps(col));
            }
#else
            for (int u = 0; u < 8; u++) {
                for (int t = 0; t < 8; t++) {
                    c[(ri + t) + (size_t)(rj + u) * ldc] = buf[(ri + t) * buf_stride + (rj + u)];
                }
            }
#endif
        }
        // ragged j tail
        for (; rj < n_src1; rj++) {
            for (int t = 0; t < 8; t++) {
                c[(ri + t) + (size_t)rj * ldc] = buf[(ri + t) * buf_stride + rj];
            }
        }
    }
    // ragged i tail
    for (; ri < n_src0; ri++) {
        for (int j = 0; j < n_src1; j++) {
            c[ri + (size_t)j * ldc] = buf[ri * buf_stride + j];
        }
    }
}