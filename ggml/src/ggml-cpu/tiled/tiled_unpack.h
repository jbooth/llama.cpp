#pragma once

// Per-format composed unpack helpers.
// Arch-neutral: extract the right pointers from the block struct, delegate
// to the arch-specific primitives (tiled_unpk_nib4, tiled_unpk_or, etc.).
//
// Categories:
// - tiled_unpk_scales: decode scales (and mins) from the packed format
// - tiled_unpk_q64<G>: decode one 64- or 32-element group in natural element order
// - tiled_gemv_scales_and_mins: unified GEMV decode (scales, mins, d, dmin)
//
// Used by both the GEMM driver (tiled_unpack_src0 in tiled.cpp) and the
// GEMV kernel (tiled_gemv_row in tiled-kernel.cpp).

#include "tiled-kernel.h"
#include "simd-mappings.h"
#include <cstring>

// LUT tables for IQ types (impl section of ggml-common.h)
#ifndef GGML_COMMON_IMPL
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#endif

// --- q4_K / q5_K: 12B packed scale+min (kmask trick) ---
static inline void tiled_unpk_scales(const block_q4_K & x, uint8_t * s8, uint8_t * m8) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    memcpy(utmp, x.scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    memcpy(s8, &utmp[0], 8);
    memcpy(m8, &utmp[2], 8);
}
static inline void tiled_unpk_scales(const block_q5_K & x, uint8_t * s8, uint8_t * m8) {
    tiled_unpk_scales(*(const block_q4_K *) &x, s8, m8);
}

// --- q6_K: 16 plain int8 scales ---
static inline void tiled_unpk_scales(const block_q6_K & x, uint8_t * s16) {
    memcpy(s16, x.scales, 16);
}

// --- q3_K: 12B packed, 6-bit scales (different kmask from q4_K) ---
static inline void tiled_unpk_scales(const block_q3_K & x, uint8_t * s16) {
    static const uint32_t kmask1 = 0x03030303;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t auxs[4];
    memcpy(auxs, x.scales, 12);
    const uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    memcpy(s16, auxs, 16);
}

// --- q2_K: 16 bytes, low 4 = scale, high 4 = min ---
static inline void tiled_unpk_scales(const block_q2_K & x, uint8_t * s16, uint8_t * m16) {
    for (int i = 0; i < 16; i++) {
        s16[i] = x.scales[i] & 0xF;
        m16[i] = x.scales[i] >> 4;
    }
}

// --- Code unpack: one 32-element group in natural element order ---
// G = group index (0..7) within the 256-K block.
//
// q4_K: G = 0..3 (64-element groups)
template <int G>
static inline void tiled_unpk_q64(const block_q4_K & x, int8_t * lo, int8_t * hi) {
    tiled_unpk_nib4(x.qs + G * 32, (uint8_t *) lo, (uint8_t *) hi);
}
// q5_K: G = 0..3 (64-element groups), qh adds 1 bit per element
template <int G>
static inline void tiled_unpk_q64(const block_q5_K & x, int8_t * lo, int8_t * hi) {
    tiled_unpk_nib4(x.qs + G * 32, (uint8_t *) lo, (uint8_t *) hi);
    tiled_unpk_or<2*G,   4, 1>((uint8_t *) lo, x.qh);
    tiled_unpk_or<2*G+1, 4, 1>((uint8_t *) hi, x.qh);
}

// q6_K: G = 0..7 (32-element groups), 4-bit ql + 2-bit qh
// ql layout: 64 bytes per 128-element half. lo nibble = first 32 elems, hi = next 32.
// qh layout: 32 bytes per 128-element half. 4 2-bit values per byte.
// Group G: ql byte offset = (G&4)*16 + (G&2)*16, use lo if G<2 or 4<G<6, else hi.
// qh byte offset = (G&4)*8, bit shift = (G&3)*2.
template <int G>
static inline void tiled_unpk_q64(const block_q6_K & x, int8_t * code) {
    constexpr int QL_OFF = (G & 4) * 16 + (G & 1) * 32;
    constexpr int QH_OFF = (G & 4) * 8;
    constexpr int BIT_S  = (G & 3) * 2;
    constexpr int IS_HI  = (G & 2) != 0;
    const uint8_t * ql = x.ql + QL_OFF;
    const uint8_t * qh = x.qh + QH_OFF;
    uint8_t lo[32], hi[32];
    tiled_unpk_nib4(ql, lo, hi);
    tiled_unpk_or<BIT_S, 4, 3>(IS_HI ? hi : lo, qh);
    memcpy(code, IS_HI ? hi : lo, 32);
}

// q3_K: G = 0..7 (32-element groups), 2-bit qs + 1-bit hmask
// G=0..3: qs[0..31], hmask bits 0..3
// G=4..7: qs[32..63], hmask bits 4..7
template <int G>
static inline void tiled_unpk_q64(const block_q3_K & x, int8_t * code) {
    constexpr int QS_OFF   = (G & 4) * 8;
    constexpr int BIT_SHIFT = (G & 3) * 2;
    tiled_unpk_2bit<BIT_SHIFT>(x.qs + QS_OFF, (uint8_t *) code);
    tiled_unpk_or<G, 2, 1>((uint8_t *) code, x.hmask);
}

// q2_K: G = 0..7 (32-element groups), 2-bit qs
template <int G>
static inline void tiled_unpk_q64(const block_q2_K & x, int8_t * code) {
    constexpr int QS_OFF    = (G & 4) * 8;
    constexpr int BIT_SHIFT = (G & 3) * 2;
    tiled_unpk_2bit<BIT_SHIFT>(x.qs + QS_OFF, (uint8_t *) code);
}

// --- Unified GEMV decode: fills 16 int16 scales, 16 int16 mins, d, dmin ---
// SUBBLK=32 types: sc[0..7], mn[0..7] used; rest zero.
// SUBBLK=16 types: all 16 used.
// BIAS types: mn[] = 0, dmin = d (kernel uses dmin for BIAS correction).
// q3_K: sc[] pre-subtracted by 32.

static inline void tiled_gemv_scales_and_mins(const block_q4_K & x, int16_t * sc, int16_t * mn, float * d, float * dmin) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    memcpy(utmp, x.scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
#if defined(__AVX2__)
    // utmp[0] = [s3,s2,s1,s0] 6-bit in byte-aligned positions, utmp[1] = [s7..s4], etc.
    _mm_storeu_si128((__m128i *) sc, _mm_cvtepi8_epi16(_mm_set_epi32(0, 0, (int)utmp[1], (int)utmp[0])));
    _mm_storeu_si128((__m128i *) mn, _mm_cvtepi8_epi16(_mm_set_epi32(0, 0, (int)utmp[3], (int)utmp[2])));
    _mm_storeu_si128((__m128i *) (sc + 8), _mm_setzero_si128());
    _mm_storeu_si128((__m128i *) (mn + 8), _mm_setzero_si128());
#else
    for (int i = 0; i < 4; i++) {
        sc[i]     = (int16_t) (utmp[0] >> (i * 8));
        sc[i + 4] = (int16_t) (utmp[1] >> (i * 8));
        mn[i]     = (int16_t) (utmp[2] >> (i * 8));
        mn[i+4]   = (int16_t) (utmp[3] >> (i * 8));
    }
    for (int i = 8; i < 16; i++) { sc[i] = 0; mn[i] = 0; }
#endif
    *d    = GGML_CPU_FP16_TO_FP32(x.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
    *dmin = GGML_CPU_FP16_TO_FP32(x.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
}

static inline void tiled_gemv_scales_and_mins(const block_q5_K & x, int16_t * sc, int16_t * mn, float * d, float * dmin) {
    tiled_gemv_scales_and_mins(*(const block_q4_K *) &x, sc, mn, d, dmin);
}

static inline void tiled_gemv_scales_and_mins(const block_q6_K & x, int16_t * sc, int16_t * mn, float * d, float * dmin) {
    for (int i = 0; i < 16; i++) { sc[i] = (int16_t)(int8_t) x.scales[i]; mn[i] = 0; }
    *d    = GGML_CPU_FP16_TO_FP32(x.d);
    *dmin = 0.0f;
}

static inline void tiled_gemv_scales_and_mins(const block_q2_K & x, int16_t * sc, int16_t * mn, float * d, float * dmin) {
    for (int i = 0; i < 16; i++) {
        sc[i] = (int16_t) (x.scales[i] & 0xF);
        mn[i] = (int16_t) (int8_t) (x.scales[i] >> 4);
    }
    *d    = GGML_CPU_FP16_TO_FP32(x.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
    *dmin = GGML_CPU_FP16_TO_FP32(x.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
}

static inline void tiled_gemv_scales_and_mins(const block_q3_K & x, int16_t * sc, int16_t * mn, float * d, float * dmin) {
    uint8_t s16[16];
    tiled_unpk_scales(x, s16);
    for (int i = 0; i < 16; i++) { sc[i] = (int16_t) s16[i] - 32; mn[i] = 0; }
    *d    = GGML_CPU_FP16_TO_FP32(x.d);
    *dmin = 0.0f;
}

// =====================================================================
// IQ types
// =====================================================================

// --- iq4_xs: 6-bit scales per 32, stored as (ls - 32) ---
static inline void tiled_unpk_scales(const block_iq4_xs & x, int8_t * s8) {
    for (int s = 0; s < 8; s++) {
        const int ls = ((x.scales_l[s/2] >> 4*(s%2)) & 0xf) | (((x.scales_h >> 2*s) & 3) << 4);
        s8[s] = (int8_t)(ls - 32);
    }
}

// --- iq2_xxs: 2-bit scales per 32, 2*bit+1 ---
static inline void tiled_unpk_scales(const block_iq2_xxs & x, uint8_t * s8) {
    const uint8_t * qs = (const uint8_t *) x.qs;
    for (int s = 0; s < 8; s++) {
        s8[s] = (uint8_t)(2 * ((qs[8*s+7] >> 4)) + 1);
    }
}

// --- iq2_xs / iq2_s: 2-bit scales per 16 (two per 32), 2*nibble+1 ---
static inline void tiled_unpk_scales(const block_iq2_xs & x, uint8_t * s16) {
    for (int s = 0; s < 8; s++) {
        s16[2*s]   = (uint8_t)(2 * (x.scales[s] & 0xf) + 1);
        s16[2*s+1] = (uint8_t)(2 * (x.scales[s] >> 4) + 1);
    }
}
static inline void tiled_unpk_scales(const block_iq2_s & x, uint8_t * s16) {
    for (int s = 0; s < 8; s++) {
        s16[2*s]   = (uint8_t)(2 * (x.scales[s] & 0xf) + 1);
        s16[2*s+1] = (uint8_t)(2 * (x.scales[s] >> 4) + 1);
    }
}

// --- iq3_xxs: 2-bit scales per 32, 2*bit+1 ---
static inline void tiled_unpk_scales(const block_iq3_xxs & x, uint8_t * s8) {
    for (int s = 0; s < 8; s++) {
        const uint32_t aux32 = *(const uint32_t *)(x.qs + QK_K/4 + 4*s);
        s8[s] = (uint8_t)(2 * (aux32 >> 28) + 1);
    }
}

// --- iq3_s: 2-bit scales per 32 (two per byte), 1+2*nibble ---
static inline void tiled_unpk_scales(const block_iq3_s & x, uint8_t * s8) {
    for (int s = 0; s < 4; s++) {
        s8[2*s]   = (uint8_t)(1 + 2 * (x.scales[s] & 0xf));
        s8[2*s+1] = (uint8_t)(1 + 2 * (x.scales[s] >> 4));
    }
}

// --- iq1_s: 3-bit scales per 32, 2*3bit+1 ---
static inline void tiled_unpk_scales(const block_iq1_s & x, uint8_t * s8) {
    for (int s = 0; s < 8; s++) {
        const uint16_t hw = x.qh[s];
        s8[s] = (uint8_t)(2 * ((hw >> 12) & 7) + 1);
    }
}

// --- iq1_m: 3-bit scales per 16 (two per 32), 2*3bit+1 ---
static inline void tiled_unpk_scales(const block_iq1_m & x, uint8_t * s16) {
    const uint16_t * sc = (const uint16_t *) x.scales;
    for (int ib = 0; ib < 8; ib++) {
        const uint16_t hw = sc[ib / 2];
        const int sh = 6 * (ib % 2);
        s16[2*ib]   = (uint8_t)(2 * ((hw >> sh) & 7) + 1);
        s16[2*ib+1] = (uint8_t)(2 * ((hw >> (sh+3)) & 7) + 1);
    }
}

// --- IQ code unpack: natural element order ---
// SUBBLK=32 types: G = 0..3, fills lo[32] (subblock 2G) + hi[32] (subblock 2G+1)
// SUBBLK=16 types: G = 0..7, fills code[32] (subblock G)

// iq4_xs: 4-bit codes through kvalues LUT (+128 bias)
template <int G>
static inline void tiled_unpk_q64(const block_iq4_xs & x, int8_t * lo, int8_t * hi) {
    static const uint8_t lut[16] = {
        (uint8_t)(kvalues_iq4nl[0] + 128),  (uint8_t)(kvalues_iq4nl[1] + 128),
        (uint8_t)(kvalues_iq4nl[2] + 128),  (uint8_t)(kvalues_iq4nl[3] + 128),
        (uint8_t)(kvalues_iq4nl[4] + 128),  (uint8_t)(kvalues_iq4nl[5] + 128),
        (uint8_t)(kvalues_iq4nl[6] + 128),  (uint8_t)(kvalues_iq4nl[7] + 128),
        (uint8_t)(kvalues_iq4nl[8] + 128),  (uint8_t)(kvalues_iq4nl[9] + 128),
        (uint8_t)(kvalues_iq4nl[10] + 128), (uint8_t)(kvalues_iq4nl[11] + 128),
        (uint8_t)(kvalues_iq4nl[12] + 128), (uint8_t)(kvalues_iq4nl[13] + 128),
        (uint8_t)(kvalues_iq4nl[14] + 128), (uint8_t)(kvalues_iq4nl[15] + 128),
    };
    uint8_t lo_nib[32], hi_nib[32];
    tiled_unpk_nib4(x.qs + 32 * G, lo_nib, hi_nib);
    tiled_lut8(lut, lo_nib + 0,  (uint8_t *) lo);
    tiled_lut8(lut, hi_nib + 0,  (uint8_t *) (lo + 16));
    tiled_lut8(lut, lo_nib + 16, (uint8_t *) hi);
    tiled_lut8(lut, hi_nib + 16, (uint8_t *) (hi + 16));
}

// iq2_xxs: 2-bit grid through iq2xxs_grid LUT + signs
template <int G>
static inline void tiled_unpk_q64(const block_iq2_xxs & x, int8_t * lo, int8_t * hi) {
    const uint8_t * qs = (const uint8_t *) x.qs;
    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *) aux32;
    uint64_t g[4];
    uint8_t signs4[4];
    memcpy(aux32, qs + 8 * (2*G), 8);
    for (int l = 0; l < 4; l++) {
        g[l] = iq2xxs_grid[aux8[l]];
        signs4[l] = ksigns_iq2xs[(aux32[1] >> (7*l)) & 127];
    }
    tiled_unpk_sign32(g[0], g[1], g[2], g[3], signs4, (uint8_t *) lo);
    memcpy(aux32, qs + 8 * (2*G+1), 8);
    for (int l = 0; l < 4; l++) {
        g[l] = iq2xxs_grid[aux8[l]];
        signs4[l] = ksigns_iq2xs[(aux32[1] >> (7*l)) & 127];
    }
    tiled_unpk_sign32(g[0], g[1], g[2], g[3], signs4, (uint8_t *) hi);
}

// iq2_xs: 2-bit grid through iq2xs_grid LUT + signs
template <int G>
static inline void tiled_unpk_q64(const block_iq2_xs & x, int8_t * code) {
    const uint16_t * q = x.qs + 4 * G;
    uint64_t g[4];
    uint8_t signs4[4];
    for (int l = 0; l < 4; l++) {
        g[l] = iq2xs_grid[q[l] & 511];
        signs4[l] = ksigns_iq2xs[q[l] >> 9];
    }
    tiled_unpk_sign32(g[0], g[1], g[2], g[3], signs4, (uint8_t *) code);
}

// iq2_s: 2-bit grid through iq2s_grid LUT + signs
template <int G>
static inline void tiled_unpk_q64(const block_iq2_s & x, int8_t * code) {
    const uint8_t * qs = x.qs + 4 * G;
    const uint8_t * signs = x.qs + QK_K/8 + 4 * G;
    uint64_t g[4];
    for (int l = 0; l < 4; l++) {
        g[l] = iq2s_grid[qs[l] | (x.qh[G] << (8 - 2*l) & 0x300)];
    }
    tiled_unpk_sign32(g[0], g[1], g[2], g[3], signs, (uint8_t *) code);
}

// iq3_xxs: 3-bit grid through iq3xxs_grid LUT + signs
template <int G>
static inline void tiled_unpk_q64(const block_iq3_xxs & x, int8_t * lo, int8_t * hi) {
    const uint8_t * qs = x.qs + 8 * (2*G);
    const uint8_t * ss = x.qs + QK_K/4 + 4 * (2*G);
    uint32_t aux32;
    uint64_t g[4];
    uint8_t signs4[4];
    memcpy(&aux32, ss, 4);
    for (int l = 0; l < 4; l++) {
        g[l] = (uint64_t) iq3xxs_grid[qs[2*l+1]] << 32 | iq3xxs_grid[qs[2*l]];
        signs4[l] = ksigns_iq2xs[(aux32 >> (7*l)) & 127];
    }
    tiled_unpk_sign32(g[0], g[1], g[2], g[3], signs4, (uint8_t *) lo);
    memcpy(&aux32, ss + 4, 4);
    for (int l = 0; l < 4; l++) {
        g[l] = (uint64_t) iq3xxs_grid[qs[2*l+1+8]] << 32 | iq3xxs_grid[qs[2*l+0+8]];
        signs4[l] = ksigns_iq2xs[(aux32 >> (7*l)) & 127];
    }
    tiled_unpk_sign32(g[0], g[1], g[2], g[3], signs4, (uint8_t *) hi);
}

// iq3_s: 3-bit grid through iq3s_grid LUT + signs
template <int G>
static inline void tiled_unpk_q64(const block_iq3_s & x, int8_t * lo, int8_t * hi) {
    const uint8_t * qs = x.qs + 8 * (2*G);
    const uint8_t * qh = x.qh + 2 * G;
    const uint8_t * signs = x.signs + 4 * (2*G);
    uint64_t g[4];
    for (int l = 0; l < 4; l++) {
        g[l] = (uint64_t) iq3s_grid[qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)] << 32
             |  iq3s_grid[qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)];
    }
    tiled_unpk_sign32(g[0], g[1], g[2], g[3], signs, (uint8_t *) lo);
    for (int l = 0; l < 4; l++) {
        g[l] = (uint64_t) iq3s_grid[qs[2*l+1+8] | ((qh[1] << (7-2*l)) & 256)] << 32
             |  iq3s_grid[qs[2*l+0+8] | ((qh[1] << (8-2*l)) & 256)];
    }
    tiled_unpk_sign32(g[0], g[1], g[2], g[3], signs + 4, (uint8_t *) hi);
}

// iq1_s: ternary grid through iq1s_grid + delta
template <int G>
static inline void tiled_unpk_q64(const block_iq1_s & x, int8_t * lo, int8_t * hi) {
    {
        const uint16_t hw = x.qh[2*G];
        const int8_t delta = (hw & 0x8000) ? -1 : 1;
        const uint8_t * qs = x.qs + 4 * (2*G);
        for (int l = 0; l < 4; l++) {
            const uint64_t entry = iq1s_grid[qs[l] | (((hw >> (3*l)) & 7) << 8)];
            tiled_unpk_tern8((const uint8_t *) &entry, delta, (uint8_t *) lo + 8*l);
        }
    }
    {
        const uint16_t hw = x.qh[2*G+1];
        const int8_t delta = (hw & 0x8000) ? -1 : 1;
        const uint8_t * qs = x.qs + 4 * (2*G+1);
        for (int l = 0; l < 4; l++) {
            const uint64_t entry = iq1s_grid[qs[l] | (((hw >> (3*l)) & 7) << 8)];
            tiled_unpk_tern8((const uint8_t *) &entry, delta, (uint8_t *) hi + 8*l);
        }
    }
}

// iq1_m: ternary grid through iq1s_grid + delta
template <int G>
static inline void tiled_unpk_q64(const block_iq1_m & x, int8_t * code) {
    const uint8_t * qs = x.qs + 4 * G;
    const uint8_t * qh = x.qh + 2 * G;
    const uint16_t idx[4] = {
        (uint16_t) (qs[0] | ((qh[0] << 8) & 0x700)),
        (uint16_t) (qs[1] | ((qh[0] << 4) & 0x700)),
        (uint16_t) (qs[2] | ((qh[1] << 8) & 0x700)),
        (uint16_t) (qs[3] | ((qh[1] << 4) & 0x700)),
    };
    const int8_t delta[4] = {
        (int8_t) ((qh[0] & 0x08) ? -1 : 1), (int8_t) ((qh[0] & 0x80) ? -1 : 1),
        (int8_t) ((qh[1] & 0x08) ? -1 : 1), (int8_t) ((qh[1] & 0x80) ? -1 : 1),
    };
    for (int l = 0; l < 4; l++) {
        const uint64_t entry = iq1s_grid[idx[l]];
        tiled_unpk_tern8((const uint8_t *) &entry, delta[l], (uint8_t *) code + 8*l);
    }
}

// --- IQ GEMV decode: fills 16 int16 scales, 16 int16 mins, d, dmin ---
// All IQ types: HAS_MIN = false, BIAS = 128, mn[] = 0, dmin = d (for BIAS correction).

static inline void tiled_gemv_scales_and_mins(const block_iq4_xs & x, int16_t * sc, int16_t * mn, float * d, float * dmin) {
    int8_t s8[8];
    tiled_unpk_scales(x, s8);
    for (int i = 0; i < 8; i++) { sc[i] = (int16_t) s8[i]; mn[i] = 0; }
    for (int i = 8; i < 16; i++) { sc[i] = 0; mn[i] = 0; }
    *d    = GGML_CPU_FP16_TO_FP32(x.d);
    *dmin = *d;
}

static inline void tiled_gemv_scales_and_mins(const block_iq2_xxs & x, int16_t * sc, int16_t * mn, float * d, float * dmin) {
    uint8_t s8[8];
    tiled_unpk_scales(x, s8);
    for (int i = 0; i < 8; i++) { sc[i] = (int16_t) s8[i]; mn[i] = 0; }
    for (int i = 8; i < 16; i++) { sc[i] = 0; mn[i] = 0; }
    *d    = GGML_CPU_FP16_TO_FP32(x.d) * 0.125f;
    *dmin = *d;
}

static inline void tiled_gemv_scales_and_mins(const block_iq2_xs & x, int16_t * sc, int16_t * mn, float * d, float * dmin) {
    uint8_t s16[16];
    tiled_unpk_scales(x, s16);
    for (int i = 0; i < 16; i++) { sc[i] = (int16_t) s16[i]; mn[i] = 0; }
    *d    = GGML_CPU_FP16_TO_FP32(x.d) * 0.125f;
    *dmin = *d;
}

static inline void tiled_gemv_scales_and_mins(const block_iq2_s & x, int16_t * sc, int16_t * mn, float * d, float * dmin) {
    uint8_t s16[16];
    tiled_unpk_scales(x, s16);
    for (int i = 0; i < 16; i++) { sc[i] = (int16_t) s16[i]; mn[i] = 0; }
    *d    = GGML_CPU_FP16_TO_FP32(x.d) * 0.125f;
    *dmin = *d;
}

static inline void tiled_gemv_scales_and_mins(const block_iq3_xxs & x, int16_t * sc, int16_t * mn, float * d, float * dmin) {
    uint8_t s8[8];
    tiled_unpk_scales(x, s8);
    for (int i = 0; i < 8; i++) { sc[i] = (int16_t) s8[i]; mn[i] = 0; }
    for (int i = 8; i < 16; i++) { sc[i] = 0; mn[i] = 0; }
    *d    = GGML_CPU_FP16_TO_FP32(x.d) * 0.25f;
    *dmin = *d;
}

static inline void tiled_gemv_scales_and_mins(const block_iq3_s & x, int16_t * sc, int16_t * mn, float * d, float * dmin) {
    uint8_t s8[8];
    tiled_unpk_scales(x, s8);
    for (int i = 0; i < 8; i++) { sc[i] = (int16_t) s8[i]; mn[i] = 0; }
    for (int i = 8; i < 16; i++) { sc[i] = 0; mn[i] = 0; }
    *d    = GGML_CPU_FP16_TO_FP32(x.d);
    *dmin = *d;
}

static inline void tiled_gemv_scales_and_mins(const block_iq1_s & x, int16_t * sc, int16_t * mn, float * d, float * dmin) {
    uint8_t s8[8];
    tiled_unpk_scales(x, s8);
    for (int i = 0; i < 8; i++) { sc[i] = (int16_t) s8[i]; mn[i] = 0; }
    for (int i = 8; i < 16; i++) { sc[i] = 0; mn[i] = 0; }
    *d    = GGML_CPU_FP16_TO_FP32(x.d) * 0.125f;
    *dmin = *d;
}

static inline void tiled_gemv_scales_and_mins(const block_iq1_m & x, int16_t * sc, int16_t * mn, float * d, float * dmin) {
    uint8_t s16[16];
    tiled_unpk_scales(x, s16);
    for (int i = 0; i < 16; i++) { sc[i] = (int16_t) s16[i]; mn[i] = 0; }
    iq1m_scale_t scale;
    const uint16_t * sc16 = (const uint16_t *) x.scales;
    scale.u16 = (sc16[0] >> 12) | ((sc16[1] >> 8) & 0x00f0) | ((sc16[2] >> 4) & 0x0f00) | (sc16[3] & 0xf000);
    *d    = GGML_CPU_FP16_TO_FP32(scale.f16) * 0.125f;
    *dmin = *d;
}
