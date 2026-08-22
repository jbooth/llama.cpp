// Tiled matmul kernel, phase 1/2: q4_K and q5_K weights x q8_K activations.
// The kernel bodies are templated on the A tile config (identical for q4_K/q5_K).
// AVX2 fallback for the microtile kernel is deferred.

#include "tiled-kernel.h"
#include "ggml-cpu-impl.h"

#include <string.h>

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#include <immintrin.h>
#endif

// 12-byte packed scale/min layout of the K-quants, unpack as in the reference kernels
#define TILED_KMASK1 0x3f3f3f3f
#define TILED_KMASK2 0x0f0f0f0f
#define TILED_KMASK3 0x03030303

void tiled_unpack_a_q4_K(const block_q4_K * rows, int64_t row_stride, int n_rows, tiled_tile_a_q4_K * tile) {
    for (int r = 0; r < n_rows; r++) {
        const block_q4_K & xb = rows[r * row_stride];

        tile->d[r]    = ggml_fp16_to_fp32(xb.d);
        tile->dmin[r] = ggml_fp16_to_fp32(xb.dmin);

        uint32_t utmp[4];
        memcpy(utmp, xb.scales, 12);
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
        for (int j = 0; j < 4; j++) {
            const uint8_t * qs = xb.qs + 32 * j;
            for (int l = 0; l < 32; l++) {
                q[64 * j + l]       = qs[l] & 0xF;
                q[64 * j + 32 + l]  = qs[l] >> 4;
            }
        }
    }
}

void tiled_unpack_a_q5_K(const block_q5_K * rows, int64_t row_stride, int n_rows, tiled_tile_a_q5_K * tile) {
    for (int r = 0; r < n_rows; r++) {
        const block_q5_K & xb = rows[r * row_stride];

        tile->d[r]    = ggml_fp16_to_fp32(xb.d);
        tile->dmin[r] = ggml_fp16_to_fp32(xb.dmin);

        uint32_t utmp[4];
        memcpy(utmp, xb.scales, 12);
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
        // 64-element chunk j uses qh bits 2j (low 32) and 2j+1 (high 32)
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        for (int j = 0; j < 4; j++) {
            const int shift = 2 * j;
            const uint8_t * ql = xb.qs + 32 * j;
            for (int l = 0; l < 32; l++) {
                const uint8_t b = xb.qh[l];
                q[64 * j + l]       = (uint8_t) ((ql[l] & 0xF) + ((b >> shift) & 1 ? 16 : 0));
                q[64 * j + 32 + l]  = (uint8_t) ((ql[l] >> 4) + ((b >> (shift + 1)) & 1 ? 16 : 0));
            }
        }
    }
}

void tiled_unpack_a_q6_K(const block_q6_K * rows, int64_t row_stride, int n_rows, tiled_tile_a_q6_K * tile) {
    constexpr int NB = TILED_TILE_K / 16; // 16-wide subblocks, 16 per 256-K
    for (int r = 0; r < n_rows; r++) {
        const block_q6_K & xb = rows[r * row_stride];
        tile->d[r] = ggml_fp16_to_fp32(xb.d);

        // 6-bit code = 4 low bits (ql) | 2 high bits (qh); see ggml_vec_dot_q6_K_q8_K_generic
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        for (int half = 0; half < 2; half++) {
            const uint8_t * ql = xb.ql + 64 * half;
            const uint8_t * qh = xb.qh + 32 * half;
            for (int l = 0; l < 32; l++) {
                q[128 * half + l +   0] = (uint8_t) ((ql[l]     & 0xF) | (((qh[l] >> 0) & 3) << 4));
                q[128 * half + l +  32] = (uint8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4));
                q[128 * half + l +  64] = (uint8_t) ((ql[l]     >>  4) | (((qh[l] >> 4) & 3) << 4));
                q[128 * half + l +  96] = (uint8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4));
            }
        }
        // scale is a plain int8 per 16-element subblock (16 per 256-K)
        for (int s = 0; s < NB; s++)
            tile->sc[r * NB + s] = (int8_t) xb.scales[s];
    }
}

void tiled_unpack_a_q3_K(const block_q3_K * rows, int64_t row_stride, int n_rows, tiled_tile_a_q3_K * tile) {
    constexpr int NB = TILED_TILE_K / 16;
    for (int r = 0; r < n_rows; r++) {
        const block_q3_K & xb = rows[r * row_stride];
        tile->d[r] = ggml_fp16_to_fp32(xb.d);

        // 3-bit code = 2 low bits (qs) | (1 high bit from hmask << 2)
        // element e (0..255): half=e>>7, el=e&127, group=el>>5, l=el&31
        //   low2 = (qs[half*32 + l] >> 2*group) & 3
        //   high = (hmask[l] >> (half*4 + group)) & 1
        // see ggml_vec_dot_q3_K_q8_K_generic
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        for (int half = 0; half < 2; half++) {
            const uint8_t * q3 = xb.qs + 32 * half;
            for (int group = 0; group < 4; group++) {
                const int bit = (half << 2) + group; // hmask bit index
                for (int l = 0; l < 32; l++) {
                    const int el = half * 128 + group * 32 + l;
                    const uint8_t low2 = (q3[l] >> (group << 1)) & 3;
                    const uint8_t high = (xb.hmask[l] >> bit) & 1;
                    q[el] = (uint8_t) (low2 | (high << 2));
                }
            }
        }
        // 6-bit scale decode (same kmask trick as the reference), stored as (sc - 32)
        uint32_t auxs[4];
        memcpy(auxs, xb.scales, 12);
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

void tiled_unpack_a_q2_K(const block_q2_K * rows, int64_t row_stride, int n_rows, tiled_tile_a_q2_K * tile) {
    constexpr int NB = TILED_TILE_K / 16;
    for (int r = 0; r < n_rows; r++) {
        const block_q2_K & xb = rows[r * row_stride];
        tile->d[r]    = ggml_fp16_to_fp32(xb.d);
        tile->dmin[r] = ggml_fp16_to_fp32(xb.dmin);

        // 2-bit code: element e -> half=e>>7, el=e&127
        //   byte = half*32 + (el & 31), shift = 2*(el >> 5)
        // see ggml_vec_dot_q2_K_q8_K_generic
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        for (int e = 0; e < TILED_TILE_K; e++) {
            const int half = e >> 7;
            const int el   = e & 127;
            q[e] = (uint8_t) ((xb.qs[(half << 5) | (el & 31)] >> ((el >> 5) << 1)) & 3);
        }
        // scale/min packed in one byte per 16-element subblock: low 4 bits = scale, high 4 = min
        for (int s = 0; s < NB; s++) {
            tile->sc[r * NB + s] = (int8_t) (xb.scales[s] & 0xF);
            tile->mn[r * NB + s] = (int8_t) (xb.scales[s] >> 4);
        }
    }
}

void tiled_unpack_b_q8_K(const block_q8_K * rows, int64_t row_stride, int n_rows, tiled_tile_b * tile) {
    const int n_padded = (n_rows + TILED_MICRO - 1) & ~(TILED_MICRO - 1);
    const int n_qv = TILED_TILE_K / 4;
    for (int r = 0; r < n_padded; r++) {
        if (r < n_rows) {
            const block_q8_K & xb = rows[r * row_stride];
            memcpy(&tile->q[r * TILED_TILE_K], xb.qs, TILED_TILE_K);
            // bsums is s-major: one contiguous 32B vector per (s, 16-row j-tile) for the kernel combine
            for (int s = 0; s < TILED_TILE_K / 16; s++) {
                tile->bsums[s * TILED_TILE_ROWS + r] = xb.bsums[s];
            }
            tile->d[r] = xb.d;
            // VNNI interleaved copy: [k/4][row][4], a 32B-aligned row-tile load per k/4 group
            const int8_t * qs = xb.qs;
            for (int g = 0; g < n_qv; g++) {
                memcpy(&tile->qv[g * TILED_TILE_ROWS * 4 + r * 4], qs + g * 4, 4);
            }
        } else {
            memset(&tile->q[r * TILED_TILE_K], 0, TILED_TILE_K);
            for (int s = 0; s < TILED_TILE_K / 16; s++) {
                tile->bsums[s * TILED_TILE_ROWS + r] = 0;
            }
            tile->d[r] = 0.0f;
            for (int g = 0; g < n_qv; g++) {
                memset(&tile->qv[g * TILED_TILE_ROWS * 4 + r * 4], 0, 4);
            }
        }
    }
}

template <typename T>
static void tiled_run_window_scalar(const T & a, const tiled_tile_b & b,
                                    int n_a, int n_b, float * buf, int buf_stride) {
    constexpr int NB = T::NB;    // subblocks per 256-K block
    constexpr int SUBBLK = TILED_TILE_K / NB;
    constexpr int NS = SUBBLK / 16; // per-16 bsums per subblock
    constexpr bool HAS_MIN = T::HAS_MIN_V;
    constexpr int BIAS = T::BIAS_V;

    for (int i0 = 0; i0 < n_a; i0 += TILED_MICRO) {
        const int n_i = n_a - i0 < TILED_MICRO ? n_a - i0 : TILED_MICRO;
        for (int j0 = 0; j0 < n_b; j0 += TILED_MICRO) {
            float acc[TILED_MICRO][TILED_MICRO];
            memset(acc, 0, sizeof(acc));

            // subdots at subblock granularity over the 256-K block, exact integer math
            for (int s = 0; s < NB; s++) {
                for (int j = 0; j < TILED_MICRO; j++) {
                    const int br = j0 + j;
                    const int8_t * qb = &b.q[br * TILED_TILE_K + s * SUBBLK];
                    int32_t bs = 0;
                    for (int u = 0; u < NS; u++) {
                        bs += b.bsums[(s * NS + u) * TILED_TILE_ROWS + br];
                    }
                    const float dB = b.d[br];

                    for (int i = 0; i < n_i; i++) {
                        const int ar = i0 + i;
                        const uint8_t * qa = &a.q[ar * TILED_TILE_K + s * SUBBLK];

                        int32_t raw = 0;
                        for (int e = 0; e < SUBBLK; e++) {
                            raw += (int32_t) qa[e] * (int32_t) qb[e];
                        }

                        // BIAS: subtract BIAS*bs (B's per-subblock code sum) from the
                        // exact int raw (q3_K=4, q6_K=32; 0 otherwise, vanishes via constexpr)
                        int32_t corr = raw;
                        if constexpr (BIAS != 0)
                            corr -= BIAS * bs;
                        const int32_t sc_raw = (int32_t) a.sc[ar * NB + s] * corr;
                        if constexpr (HAS_MIN) {
                            const int32_t mn_bs = (int32_t) a.mn[ar * NB + s] * bs;
                            acc[i][j] += dB * ((float) a.d[ar] * (float) sc_raw
                                             - (float) a.dmin[ar] * (float) mn_bs);
                        } else {
                            acc[i][j] += dB * (float) a.d[ar] * (float) sc_raw;
                        }
                    }
                }
            }

            // accumulate the microtile into the j-major buffer (unconditional 16x16; the
            // ragged rows/cols past n_i/n_j hold the microtile's zero padding)
            for (int i = 0; i < TILED_MICRO; i++) {
                for (int j = 0; j < TILED_MICRO; j++) {
                    buf[(i0 + i) * buf_stride + (j0 + j)] += acc[i][j];
                }
            }
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
// A 512-bit register (the __m512 or __m512i C type, "ZMM" register in
// Intel's naming) is 64 bytes. The C type is only a hint to the
// compiler; the same 64 physical bytes are read differently depending
// on which intrinsics touch them:
//   - __m512  as 16 x float32 lanes
//   - __m512i as 16 x int32 lanes
//   - 64 x int8 only inside the dpbusd dot instruction
// The machine's register file holds 32 of these registers. This kernel
// is register-pressure-driven: the 8 int accumulators (acc16[]) + 8
// float accumulators (accf[]) + a handful of live temps must all stay
// resident in those 32 registers. If any accumulator spills to the
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
// unsigned/signed split is a hardware fact: A must hold the unsigned
// weight codes, B the signed q8 (plan section 4).
//
// Why the 16 lanes mean 16 B-columns instead of 16 k-elements: for the
// 16 lanes to be 16 different output pairs (outer-product structure),
// every lane of the A register must hold the SAME 4 A-codes (one
// A-row's codes for this 4-k group, broadcast), so that lane j pairs
// them with a different B column's 4 codes. The B register therefore
// holds 16 distinct columns. That is why B is pre-interleaved at
// unpack time into the qv layout [k/4][row][4]: for a fixed 4-k group
// kg and a run of 16 B-columns, the needed bytes are exactly one
// contiguous 64B chunk, so the B-operand is a single load that is
// reused across all 8 A-rows, and the A-operand is a cheap scalar
// broadcast.
//
// Math split (plan section 7.1). The subblock contribution to
// C[i][j] is
//     dB[j] * ( dA[i]*sc[i][s]*raw - dmin[i]*mn[i][s]*bs[j] )
// where raw is the exact integer dot of the unsigned A codes with the
// signed B codes over the SUBBLK elements, and bs[j] is the sum of the
// B codes over the same subblock (needed for the min correction). The
// int part runs on int32 accumulators that live for ONE subblock (8
// dpbusd at SUBBLK=32); the correction is applied exactly (all values
// < 2^24, exact in f32) and folded into the float accumulators, which
// live across the whole 256-K block.
//
// Microtile shape: 8 A-rows x 16 B-columns. 8, because acc16[8] +
// accf[8] + temps fit the 32-ZMM file; 16, because that is what one
// dpbusd covers in the B-column direction. A full 16x16 microtile was
// analyzed and rejected for exactly the register-count reason above.
#define TILED_VNNI_A 8
#define TILED_VNNI_B 16

template <typename T>
static void tiled_run_window_vnni(const T & a, const tiled_tile_b & b,
                                  int n_a, int n_b, float * buf, int buf_stride) {
    constexpr int NB = T::NB;    // subblocks per 256-K block
    constexpr int SUBBLK = TILED_TILE_K / NB;
    constexpr int NS = SUBBLK / 16; // per-16 bsums per subblock
    constexpr int NG = SUBBLK / 4;  // 4-k dpbusd groups per subblock (8 for q4/q5_K, 4 for the rest)
    constexpr bool HAS_MIN = T::HAS_MIN_V;
    constexpr int BIAS = T::BIAS_V;

    // buf is the j-major 256x256 float workspace shared by all k-blocks:
    // buf[i*buf_stride + j] holds the running partial sum for C[i][j]
    // over the 256-K blocks processed so far (zeroed once per window by
    // the driver). This function adds ONE 256-K block's worth of it.
    //
    // Loop nest, outer to inner:
    //   i0: bands of TILED_VNNI_A (=8) A-rows (weight rows)
    //   j0: bands of TILED_VNNI_B (=16) B-columns (activation rows,
    //       i.e. the columns of C in i-major order)
    //   s : subblocks of SUBBLK k-elements within the 256-K block
    //       (NB of them; NB=8 for q4_K/q5_K)
    //   g : groups of 4 k-elements within a subblock (SUBBLK/4 of them;
    //       one dpbusd per (g, t) pair)
    //   t : the 8 A-rows inside the i0 band
    for (int i0 = 0; i0 < n_a; i0 += TILED_VNNI_A) {
        for (int j0 = 0; j0 < n_b; j0 += TILED_VNNI_B) {

            // dB_vec: _mm512_loadu_ps loads 16 float32 (64 bytes) from
            // unaligned memory into a 512-bit register (first AVX-512 op
            // in the function). b.d[] is contiguous in j, so the 16
            // activation scales for this j-tile are one load.
            // After this: dB_vec lane j = dB[j0+j], the f32 scale of
            // B-column j0+j.
            const __m512 dB_vec = _mm512_loadu_ps(&b.d[j0]);

            // accf[t]: the float accumulators. _mm512_setzero_ps builds
            // a 512-bit register with all 16 float lanes zero.
            // After this: accf[t] all lanes 0.
            // Meaning going forward: accf[t] lane j = partial C[i0+t]
            // [j0+j] over the subblocks seen so far in this 256-K block.
            // Held across the whole block; written into buf exactly once
            // per microtile, at the very end.
            __m512 accf[TILED_VNNI_A];
            for (int t = 0; t < TILED_VNNI_A; t++) accf[t] = _mm512_setzero_ps();

            for (int s = 0; s < NB; s++) {
                // --- build bs_dB_vec: the B-side min-correction term, pre-scaled ---
                //
                // The correction needs bs[j] = sum of the B codes of
                // column j0+j over the SUBBLK elements of subblock s.
                // B's bsums table stores the sums per 16 elements, laid
                // out s-major [s16][row] (see tiled_tile_b), so a
                // subblock is NS per-16 values (NS = SUBBLK/16 = 2 for
                // q4_K/q5_K: subblock s covers per-16 groups 2s and
                // 2s+1).
                //
                // _mm256_loadu_si256 loads 32 bytes into a 256-bit
                // register (16 int16 lanes). bsums[s16][j0..j0+15] is
                // exactly 16 int16 = 32B, and because the table is
                // s-major those 16 rows are CONTIGUOUS in memory, so
                // no gather is needed. (AVX512VL, required by the build
                // guard above, is what lets 256-bit values feed 512-bit
                // ops and be used alongside each other.)
                //
                // _mm512_cvtepi16_epi32 sign-extends the 16 int16 lanes
                // of a 256-bit register into 16 int32 lanes of a 512-bit
                // register. We need int32 width before adding the NS
                // groups; max |bs| is 2*16*127 = 4064, far inside
                // int32, and the sign matters (B codes are signed).
                // After this: bs32 lane j = bsums[s*NS][j0+j], the B sum
                // over the FIRST per-16 group of the subblock.
                __m512i bs32 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *) &b.bsums[s * TILED_TILE_ROWS * NS + j0]));
                // _mm512_add_epi32 adds two 512-bit registers lane-wise:
                // 16 independent int32 adds in one instruction. For NS > 1
                // this folds in the remaining per-16 groups.
                // After the loop: bs32 lane j = bs[j0+j], the exact sum
                // of the B codes over the WHOLE subblock for column j0+j.
                for (int u = 1; u < NS; u++) {
                    bs32 = _mm512_add_epi32(bs32, _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *) &b.bsums[(s * NS + u) * TILED_TILE_ROWS + j0])));
                }
                // _mm512_cvtepi32_ps converts 16 int32 lanes to 16
                // float32 lanes (exact here: values are integers below
                // 2^24). _mm512_mul_ps multiplies 16 float lane-pairs,
                // each with its own rounding.
                // After this: bs_dB_vec lane j = bs[j0+j] * dB[j0+j],
                // the whole B-side correction factor pre-scaled by the
                // activation scale, so the per-row correction below is a
                // single fused op.
                const __m512 bs_dB_vec = _mm512_mul_ps(_mm512_cvtepi32_ps(bs32), dB_vec);

                // acc16[t]: the int accumulators, one per A-row of the
                // band. _mm512_setzero_si512 is the integer-spelling of
                // the same 16-lane zero (64 bits per lane, read as int32
                // here).
                // After this: acc16[t] all lanes 0.
                // Meaning going forward: acc16[t] lane j = raw(t, j0+j,
                // s), the exact integer dot of A-row (i0+t) with B-column
                // (j0+j) over subblock s, built up by the NG dpbusd of the
                // g-loop below. |raw| <= SUBBLK*63*127 ~ 2.6e5 worst case
                // (q6_K 6-bit code): no int32 overflow, no precision loss.
                __m512i acc16[TILED_VNNI_A];
                for (int t = 0; t < TILED_VNNI_A; t++) acc16[t] = _mm512_setzero_si512();

                for (int g = 0; g < NG; g++) {
                    const int kg = s * NG + g; // global 4-k group
                    // _mm512_loadu_si512 loads 64 bytes (16 int32 lanes)
                    // into a 512-bit register; the "u" = unaligned, and
                    // it works on aligned addresses too (ours happen to
                    // be 64B-aligned: qv base is 32B-aligned, offset
                    // kg*1024 + j0*4 is a multiple of 64).
                    // The qv interleave (built at unpack) stores, per 4-k
                    // group kg, 4 contiguous B-columns back to back:
                    // qv[kg*1024 + r*4 .. +3] = qB[r][kg*4 .. +3]. The
                    // 64 bytes loaded here are exactly the 16 columns
                    // j0..j0+15 x 4 k-codes.
                    // After this: Bz lane j = the 4 signed q8 bytes of
                    // B-column (j0+j) for k-elements kg*4 .. kg*4+3.
                    const __m512i Bz = _mm512_loadu_si512((const __m512i *) &b.qv[kg * TILED_TILE_ROWS * 4 + j0 * 4]);
                    // All A-rows computed unconditionally (rows past n_a
                    // hold harmless tile garbage; only the final store to
                    // C is clamped). The fixed trip count lets the
                    // compiler keep acc16 in the register file instead
                    // of RMW-ing the stack -- see the v1 regression note
                    // at the top.
                    for (int t = 0; t < TILED_VNNI_A; t++) {
                        // a4: the 4 unsigned weight codes of A-row (i0+t)
                        // for k-elements kg*4 .. kg*4+3, read as one
                        // 32-bit word (they are packed 1 byte each in
                        // a.q, which the unpacker built).
                        const uint32_t a4 = *(const uint32_t *) &a.q[(i0 + t) * TILED_TILE_K + kg * 4];
                        // _mm512_set1_epi32 broadcasts one 32-bit scalar
                        // into all 16 int32 lanes.
                        // After this: Ab lane j = a4 for EVERY j. That is
                        // the outer-product form from the header: the
                        // same 4 A-codes face every B column, so lane j
                        // computes the pair (A-row t, B-column j0+j).
                        const __m512i Ab = _mm512_set1_epi32((int) a4);
                        // The VNNI op itself (full semantics at the top):
                        // acc16[t][i] += sum of 4 byte-products between
                        // Ab lane i (the A codes) and Bz lane i (the B
                        // codes), unsigned x signed.
                        // After this: acc16[t] lane j = the partial
                        // integer dot of A-row (i0+t) with B-column (j0+j)
                        // over k-groups 0..g of subblock s. 4:1 -- one
                        // instruction covers 4 k-elements per output pair,
                        // no horizontal reduce needed.
                        acc16[t] = _mm512_dpbusd_epi32(acc16[t], Ab, Bz);
                    }
                }
                // --- correction: fold this subblock's exact int sum into accf ---
                //
                // After the g-loop: acc16[t] lane j = raw(t, j0+j, s),
                // complete and exact (all NG groups of 4 k-elements = the
                // full SUBBLK). The algebra says this subblock adds
                //     dB[j] * ( dA[t]*sc[t][s]*(raw - BIAS*bs[j]) - dmin[t]*mn[t][s]*bs[j] )
                // to C[i0+t][j0+j]. The BIAS term (q3_K=4, q6_K=32; 0
                // otherwise) is subtracted from the exact int raw below; the
                // dmin*mn term exists only for HAS_MIN (q4/q5/q2_K). Every
                // factor is small enough that the whole thing is exact in
                // f32 (products < 2^24). All 16 columns are converted and
                // scaled in one pass per row:
                for (int t = 0; t < TILED_VNNI_A; t++) {
                    const int ar = i0 + t;
                    // BIAS correction, exact in int32 (see plan section 4):
                    // sum(c*qB) with c = u - BIAS = sum(u*qB) - BIAS*sum(qB).
                    // bs32 is B's per-subblock code sum (per column); BIAS is a
                    // tiny scalar, so BIAS*bs32 is exact and fits int32. For the
                    // BIAS==0 formats this whole if constexpr vanishes.
                    __m512i rawi = acc16[t];
                    if constexpr (BIAS != 0)
                        rawi = _mm512_sub_epi32(rawi, _mm512_mullo_epi32(bs32, _mm512_set1_epi32(BIAS)));
                    // _mm512_cvtepi32_ps again (see the bs_dB_vec build):
                    // 16 int32 -> 16 float32, exact in range.
                    // After this: f16 lane j = (raw - BIAS*bs)(t, j0+j, s) as float.
                    const __m512 f16 = _mm512_cvtepi32_ps(rawi);
                    // _mm512_set1_ps is the float-spelling of set1:
                    // broadcast one float32 scalar into all 16 lanes. The
                    // scalar a.d[ar]*sc[ar*NB+s] = dA[t]*sc[t][s] is the
                    // per-(row, subblock) weight scale (q3_K: sc already = raw-32).
                    // After this: r lane j = dA[t]*sc[t][s]*(raw - BIAS*bs).
                    const __m512 r = _mm512_mul_ps(f16, _mm512_set1_ps(a.d[ar] * (float) a.sc[ar * NB + s]));
                    // _mm512_mul_ps again: scale on the activation side.
                    // After this: out lane j = dA[t]*sc[t][s]*(raw - BIAS*bs)*dB[j].
                    __m512 out = _mm512_mul_ps(r, dB_vec);
                    // min term, HAS_MIN formats only (q4/q5/q2_K): subtract
                    // dmin[t]*mn[t][s]*bs[j]*dB[j]. _mm512_fnmadd_ps(A, B, C)
                    // computes B*C - A with ONE rounding (negative FMA).
                    if constexpr (HAS_MIN)
                        out = _mm512_fnmadd_ps(_mm512_set1_ps(a.dmin[ar] * (float) a.mn[ar * NB + s]), bs_dB_vec, out);
                    // _mm512_add_ps: plain lane-wise float add (the
                    // non-fused sibling of fnmadd).
                    // After this: accf[t] lane j = partial C[i0+t][j0+j]
                    // over subblocks 0..s of this 256-K block.
                    accf[t] = _mm512_add_ps(accf[t], out);
                }
            }
            // --- write the microtile into the j-major buffer ---
            //
            // After the s-loop: accf[t] lane j holds the complete
            // 256-K-block partial sum for C[i0+t][j0+j] (all NB
            // subblocks). buf accumulates across k-blocks, so each of
            // the 8 rows is a read-add-store. _mm512_loadu_ps (16
            // floats, as for dB_vec) + _mm512_add_ps (as above) +
            // _mm512_storeu_ps, which stores 16 float32 (64 bytes) to
            // unaligned memory. The buffer row for this j-tile is 16
            // CONTIGUOUS floats (j-major layout), so each row is one
            // 64B-aligned add: no striding, no write amplification.
            // (Rows past n_a add harmless garbage that the driver's
            // store to C never reads.)
            for (int t = 0; t < TILED_VNNI_A; t++) {
                float * p = &buf[(i0 + t) * buf_stride + j0];
                _mm512_storeu_ps(p, _mm512_add_ps(_mm512_loadu_ps(p), accf[t]));
            }
        }
    }
}

#endif // __AVX512VNNI__ && __AVX512VL__

template <typename T>
void tiled_run_window(const T & a, const tiled_tile_b & b,
                      int n_a, int n_b, float * buf, int buf_stride) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
    tiled_run_window_vnni(a, b, n_a, n_b, buf, buf_stride);
#else
    tiled_run_window_scalar(a, b, n_a, n_b, buf, buf_stride);
#endif
}

// explicit instantiations for the in-use tile types (q4_K and q5_K share the layout)
template void tiled_run_window<tiled_tile_a_q4_K>(const tiled_tile_a_q4_K & a, const tiled_tile_b & b,
                                                  int n_a, int n_b, float * buf, int buf_stride);
template void tiled_run_window<tiled_tile_a_q6_K>(const tiled_tile_a_q6_K & a, const tiled_tile_b & b,
                                                  int n_a, int n_b, float * buf, int buf_stride);
template void tiled_run_window<tiled_tile_a_q3_K>(const tiled_tile_a_q3_K & a, const tiled_tile_b & b,
                                                  int n_a, int n_b, float * buf, int buf_stride);
template void tiled_run_window<tiled_tile_a_q2_K>(const tiled_tile_a_q2_K & a, const tiled_tile_b & b,
                                                  int n_a, int n_b, float * buf, int buf_stride);

// Transpose-store the j-major buffer to C (i contiguous). The buffer's natural layout
// matches the VNNI microtile output (j-major), so per-block accumulation is a cheap
// contiguous add; the one-shot transpose here pays for the i-major C order. 8x8 blocks
// give contiguous 32B stores to C (no write amplification).
void tiled_store_window(const float * buf, int n_a, int n_b, int buf_stride, float * c, size_t ldc) {
    int i0 = 0;
    for (; i0 + 8 <= n_a; i0 += 8) {
        int j0 = 0;
        for (; j0 + 8 <= n_b; j0 += 8) {
#if defined(__AVX2__)
            float r[8][8];
            for (int t = 0; t < 8; t++) {
                _mm256_storeu_ps(&r[t][0], _mm256_loadu_ps(&buf[(i0 + t) * buf_stride + j0]));
            }
            for (int u = 0; u < 8; u++) {
                float col[8];
                for (int t = 0; t < 8; t++) col[t] = r[t][u];
                _mm256_storeu_ps(&c[i0 + (size_t)(j0 + u) * ldc], _mm256_loadu_ps(col));
            }
#else
            for (int u = 0; u < 8; u++) {
                for (int t = 0; t < 8; t++) {
                    c[(i0 + t) + (size_t)(j0 + u) * ldc] = buf[(i0 + t) * buf_stride + (j0 + u)];
                }
            }
#endif
        }
        // ragged j tail
        for (; j0 < n_b; j0++) {
            for (int t = 0; t < 8; t++) {
                c[(i0 + t) + (size_t)j0 * ldc] = buf[(i0 + t) * buf_stride + j0];
            }
        }
    }
    // ragged i tail
    for (; i0 < n_a; i0++) {
        for (int j = 0; j < n_b; j++) {
            c[i0 + (size_t)j * ldc] = buf[i0 * buf_stride + j];
        }
    }
}