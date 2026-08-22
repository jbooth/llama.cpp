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

void tiled_unpack_b_q8_K(const block_q8_K * rows, int64_t row_stride, int n_rows, tiled_tile_b * tile) {
    const int n_padded = (n_rows + TILED_MICRO - 1) & ~(TILED_MICRO - 1);
    const int n_qv = TILED_TILE_K / 4;
    for (int r = 0; r < n_padded; r++) {
        if (r < n_rows) {
            const block_q8_K & xb = rows[r * row_stride];
            memcpy(&tile->q[r * TILED_TILE_K], xb.qs, TILED_TILE_K);
            for (int s = 0; s < TILED_TILE_K / 16; s++) {
                tile->bsums[r * (TILED_TILE_K / 16) + s] = xb.bsums[s];
            }
            tile->d[r] = xb.d;
            // VNNI interleaved copy: [k/4][row][4], a 32B-aligned row-tile load per k/4 group
            const int8_t * qs = xb.qs;
            for (int g = 0; g < n_qv; g++) {
                memcpy(&tile->qv[g * TILED_TILE_ROWS * 4 + r * 4], qs + g * 4, 4);
            }
        } else {
            memset(&tile->q[r * TILED_TILE_K], 0, TILED_TILE_K);
            memset(&tile->bsums[r * (TILED_TILE_K / 16)], 0, (TILED_TILE_K / 16) * sizeof(int16_t));
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
                        bs += b.bsums[br * 16 + s * NS + u];
                    }
                    const float dB = b.d[br];

                    for (int i = 0; i < n_i; i++) {
                        const int ar = i0 + i;
                        const uint8_t * qa = &a.q[ar * TILED_TILE_K + s * SUBBLK];

                        int32_t raw = 0;
                        for (int e = 0; e < SUBBLK; e++) {
                            raw += (int32_t) qa[e] * (int32_t) qb[e];
                        }

                        const int32_t sc_raw = (int32_t) a.sc[ar * NB + s] * raw;
                        const int32_t mn_bs  = (int32_t) a.mn[ar * NB + s] * bs;

                        acc[i][j] += dB * ((float) a.d[ar] * (float) sc_raw
                                         - (float) a.dmin[ar] * (float) mn_bs);
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

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)

// VNNI microtile kernel (register strategy):
// - one 512-bit dpbusd covers 16 B-columns x 4 k (4:1, no horizontal reduce)
// - B is pre-interleaved (qv) so each dpbusd B-operand is one 32B-aligned load,
//   loaded once per 4-k group and reused across the A-rows
// - A-operand is a set1 broadcast of the 4 k-codes of one A-row
// - int32 raw is accumulated per subblock (8 dpbusd), then the exact
//   d/sc/dmin/mn correction is applied and folded into a float accumulator held
//   across the whole 256-K block; each microtile adds into the j-major buffer
// Microtile is 8 A-rows x 16 B-cols so acc16[8] (int32) + accf[8] (float) fit
// the 32-ZMM file alongside the temps.
#define TILED_VNNI_A 8
#define TILED_VNNI_B 16

template <typename T>
static void tiled_run_window_vnni(const T & a, const tiled_tile_b & b,
                                  int n_a, int n_b, float * buf, int buf_stride) {
    constexpr int NB = T::NB;    // subblocks per 256-K block
    constexpr int SUBBLK = TILED_TILE_K / NB;
    constexpr int NS = SUBBLK / 16; // per-16 bsums per subblock

    for (int i0 = 0; i0 < n_a; i0 += TILED_VNNI_A) {
        for (int j0 = 0; j0 < n_b; j0 += TILED_VNNI_B) {

            // per j-tile: dB vector (16 lanes)
            float dbv[TILED_VNNI_B];
            for (int jj = 0; jj < TILED_VNNI_B; jj++) dbv[jj] = b.d[j0 + jj];
            const __m512 dB_vec = _mm512_loadu_ps(dbv);

            // float accumulators, held across the 256-K block (added into the buffer per microtile)
            __m512 accf[TILED_VNNI_A];
            for (int t = 0; t < TILED_VNNI_A; t++) accf[t] = _mm512_setzero_ps();

            for (int s = 0; s < NB; s++) {
                // per subblock: bsumsB[c][s] * dB[c] as a 16-lane float vector
                float bsdb[TILED_VNNI_B];
                for (int jj = 0; jj < TILED_VNNI_B; jj++) {
                    const int br = j0 + jj;
                    int32_t bs = 0;
                    for (int u = 0; u < NS; u++) {
                        bs += b.bsums[br * 16 + s * NS + u];
                    }
                    bsdb[jj] = (float) bs * b.d[br];
                }
                const __m512 bs_dB_vec = _mm512_loadu_ps(bsdb);

                // int32 raw accumulators for this subblock (16 B-cols each)
                __m512i acc16[TILED_VNNI_A];
                for (int t = 0; t < TILED_VNNI_A; t++) acc16[t] = _mm512_setzero_si512();

                for (int g = 0; g < 8; g++) {
                    const int kg = s * 8 + g; // global 4-k group
                    const __m512i Bz = _mm512_loadu_si512((const __m512i *) &b.qv[kg * TILED_TILE_ROWS * 4 + j0 * 4]);
                    // all A-rows computed unconditionally (rows past n_i hold harmless tile
                    // garbage; only the C-write below is guarded by n_i). Fixed trip count lets
                    // the compiler keep acc16 in the register file instead of RMW-ing the stack.
                    for (int t = 0; t < TILED_VNNI_A; t++) {
                        const uint32_t a4 = *(const uint32_t *) &a.q[(i0 + t) * TILED_TILE_K + kg * 4];
                        const __m512i Ab = _mm512_set1_epi32((int) a4);
                        acc16[t] = _mm512_dpbusd_epi32(acc16[t], Ab, Bz);
                    }
                }

                // exact correction, folded into the float accumulator
                for (int t = 0; t < TILED_VNNI_A; t++) {
                    const int ar = i0 + t;
                    const __m512 f16 = _mm512_cvtepi32_ps(acc16[t]);
                    const __m512 r = _mm512_mul_ps(f16, _mm512_set1_ps(a.d[ar] * (float) a.sc[ar * NB + s]));
                    __m512 out = _mm512_mul_ps(r, dB_vec);
                    out = _mm512_fnmadd_ps(_mm512_set1_ps(a.dmin[ar] * (float) a.mn[ar * NB + s]), bs_dB_vec, out);
                    accf[t] = _mm512_add_ps(accf[t], out);
                }
            }

            // accumulate the 8 A-row vectors into the j-major buffer; each accf[t] is 16
            // contiguous floats (TILED_VNNI_B), so this is one aligned 64B add per row
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
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    tiled_run_window_vnni(a, b, n_a, n_b, buf, buf_stride);
#else
    tiled_run_window_scalar(a, b, n_a, n_b, buf, buf_stride);
#endif
}

// explicit instantiations for the in-use tile types (q4_K and q5_K share the layout)
template void tiled_run_window<tiled_tile_a_q4_K>(const tiled_tile_a_q4_K & a, const tiled_tile_b & b,
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