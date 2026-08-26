
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "tiled.h"

#include "ggml-quants.h"
#include "tiled-kernel.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <new>

#define QK_K 256 // TODO why don't we pick this up properly from headers

// === src0/src1 unpack (driver space; the VNNI code fill delegates to the kernel) ===

// 32-byte-unit unpack primitives for the src0-format code expansion. All codes are
// bitfields with no overlap, so merging a flag into an already-extracted code uses OR
// (identical to ADD, matches the reference kernels' semantics).
#if defined(__AVX2__)
// packed 4-bit codes -> low nibbles (lo) + high nibbles (hi)
static inline void tiled_unpk_nib4(const uint8_t * src, uint8_t * lo, uint8_t * hi) {
    const __m256i v = _mm256_loadu_si256((const __m256i *) src);
    // mask before the lane shift so bits do not cross byte boundaries
    _mm256_storeu_si256((__m256i *) lo, _mm256_and_si256(v, _mm256_set1_epi8(0x0F)));
    _mm256_storeu_si256((__m256i *) hi, _mm256_srli_epi32(_mm256_and_si256(v, _mm256_set1_epi8((int8_t) 0xF0)), 4));
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
    for (int l = 0; l < 32; l++) { dst[l] = (uint8_t) ((src[l] >> S) & 3); }
}
template <int S, int D, int M>
static inline void tiled_unpk_or(uint8_t * dst, const uint8_t * src) {
    for (int l = 0; l < 32; l++) { dst[l] = (uint8_t) (dst[l] | (((src[l] >> S) & M) << D)); }
}
#endif

static void tiled_unpack_src0_q4_K(const block_q4_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q4_K * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = tiled_tile_src0_q4_K::NB;
    // 12-byte packed scale/min decode, same extraction as the reference kernels
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    for (int r = 0; r < n_rows; r++) {
        const block_q4_K & x = rows[r * row_stride];

        tile->d[r]    = ggml_fp16_to_fp32(x.d);
        tile->dmin[r] = ggml_fp16_to_fp32(x.dmin);

        uint32_t utmp[4];
        memcpy(utmp, x.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * scales = (const uint8_t *) &utmp[0];
        const uint8_t * mins = (const uint8_t *) &utmp[2];
        for (int s = 0; s < NB; s++) {
            tile->scales[r * NB + s] = (int32_t) scales[s];
            tile->mins[r * NB + s] = (int32_t) mins[s];
        }

        // extract the 4-bit codes (low 4 + high 4), same extraction as the reference kernels
        uint8_t * q = &tile->q[r * TILED_TILE_K];
        tiled_unpk_nib4(x.qs + 0,  q + 0,   q + 32);
        tiled_unpk_nib4(x.qs + 32, q + 64,  q + 96);
        tiled_unpk_nib4(x.qs + 64, q + 128, q + 160);
        tiled_unpk_nib4(x.qs + 96, q + 192, q + 224);
    }
}

static void tiled_unpack_src0_q5_K(const block_q5_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q5_K * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = tiled_tile_src0_q5_K::NB;
    // 12-byte packed scale/min decode, same extraction as the reference kernels
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    for (int r = 0; r < n_rows; r++) {
        const block_q5_K & x = rows[r * row_stride];

        tile->d[r]    = ggml_fp16_to_fp32(x.d);
        tile->dmin[r] = ggml_fp16_to_fp32(x.dmin);

        uint32_t utmp[4];
        memcpy(utmp, x.scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * scales = (const uint8_t *) &utmp[0];
        const uint8_t * mins = (const uint8_t *) &utmp[2];
        for (int s = 0; s < NB; s++) {
            tile->scales[r * NB + s] = (int32_t) scales[s];
            tile->mins[r * NB + s] = (int32_t) mins[s];
        }

        // extract the 5-bit codes (4 low bits + 1 high bit), same as the generic kernels:
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

static void tiled_unpack_src0_q6_K(const block_q6_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q6_K * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = tiled_tile_src0_q6_K::NB;
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
        for (int s = 0; s < NB; s++) { tile->scales[r * NB + s] = (int32_t) (int8_t) x.scales[s]; }
    }
}

static void tiled_unpack_src0_q3_K(const block_q3_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q3_K * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = tiled_tile_src0_q3_K::NB;
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
        // 6-bit scale decode (same kmask trick as the reference), stored as (scales - 32)
        static const uint32_t kmask1 = 0x03030303;
        static const uint32_t kmask2 = 0x0f0f0f0f;
        uint32_t auxs[4];
        memcpy(auxs, x.scales, 12);
        const uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t * scales = (const int8_t *) &auxs[0];
        for (int s = 0; s < NB; s++) { tile->scales[r * NB + s] = (int32_t) scales[s] - 32; }
    }
}

static void tiled_unpack_src0_q2_K(const block_q2_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q2_K * tile) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr int NB = tiled_tile_src0_q2_K::NB;
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
            tile->scales[r * NB + s] = (int32_t) (x.scales[s] & 0xF);
            tile->mins[r * NB + s] = (int32_t) (x.scales[s] >> 4);
        }
    }
}

// tag-dispatched unpack: rows is the base of one (window, k-block), cast to the format's block type
static inline void tiled_unpack_src0(tiled_fmt_q4_K, const void * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q4_K * tile) {
    tiled_unpack_src0_q4_K((const block_q4_K *) rows, row_stride, n_rows, tile);
}
static inline void tiled_unpack_src0(tiled_fmt_q5_K, const void * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q5_K * tile) {
    tiled_unpack_src0_q5_K((const block_q5_K *) rows, row_stride, n_rows, tile);
}
static inline void tiled_unpack_src0(tiled_fmt_q6_K, const void * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q6_K * tile) {
    tiled_unpack_src0_q6_K((const block_q6_K *) rows, row_stride, n_rows, tile);
}
static inline void tiled_unpack_src0(tiled_fmt_q3_K, const void * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q3_K * tile) {
    tiled_unpack_src0_q3_K((const block_q3_K *) rows, row_stride, n_rows, tile);
}
static inline void tiled_unpack_src0(tiled_fmt_q2_K, const void * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q2_K * tile) {
    tiled_unpack_src0_q2_K((const block_q2_K *) rows, row_stride, n_rows, tile);
}

// src1 tile from q8_K rows: one memcpy per row in natural [row][k] order, plus
// d and bsums; every build. On builds where the kernel reads the codes in a
// different order (KERNEL_SRC1_UNPACK, i.e. x86 VNNI), the code fill is
// delegated to the kernel's op, which copies from the up-front interleaved
// region; d and bsums stay here.
static void tiled_unpack_src1_q8_K(const block_q8_K * rows, int64_t row_stride, int n_rows, tiled_tile_src1 * tile,
                                   const int8_t * qv, int64_t nr1_pad, int64_t r_start, int64_t kblk) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    const int n_padded = (n_rows + TILED_MICRO - 1) & ~(TILED_MICRO - 1);
#if defined(KERNEL_SRC1_UNPACK)
    tiled_unpack_src1_q8_K_kernel(n_rows, tile, qv, nr1_pad, r_start, kblk);
#else
    (void) qv;
    (void) nr1_pad;
    (void) r_start;
    (void) kblk;
    for (int r = 0; r < n_padded; r++) {
        if (r < n_rows) {
            memcpy(&tile->q[r * TILED_TILE_K], rows[r * row_stride].qs, TILED_TILE_K);
        } else {
            memset(&tile->q[r * TILED_TILE_K], 0, TILED_TILE_K);
        }
    }
#endif
    // d and bsums: natural layout, every build; bsums stored int32 (one 64B
    // vector load per (s, j0) on VNNI, no per-use cvt on the AVX tiers), plan 12.9
    for (int r = 0; r < n_padded; r++) {
        if (r < n_rows) {
            const block_q8_K & x = rows[r * row_stride];
            for (int s = 0; s < TILED_TILE_K / TILED_MICRO; s++) { tile->bsums[s * TILED_TILE_ROWS + r] = (int32_t) x.bsums[s]; }
            tile->d[r] = x.d;
        } else {
            for (int s = 0; s < TILED_TILE_K / TILED_MICRO; s++) { tile->bsums[s * TILED_TILE_ROWS + r] = 0; }
            tile->d[r] = 0.0f;
        }
    }
}

// === new tiled path (K-quants, scalar + VNNI kernel) ===

// The src1 tile and the j-major float accumulator are identical for every src0 format,
// so they are shared thread_local state; only the src0 tile differs (SUBBLK changes
// the code density and the per-subblock side tables), so it is held per-format via
// tiled_fmt_tile<Fmt>. A thread processes one op (one format) at a time, so sharing
// acc/b across formats is safe: each window zeroes acc and each chunk rebuilds b.
struct tiled_kernel_ws {
    tiled_tile_src1 * src1 = nullptr;
    float        * acc = nullptr;

    ~tiled_kernel_ws() {
        delete src1;
        // acc was allocated 64B-aligned (std::align_val_t), so free with the
        // matching aligned delete, not delete[]
        if (acc) {
            ::operator delete(acc, std::align_val_t(64));
        }
    }
};

static thread_local tiled_kernel_ws tiled_ws;

template <typename Fmt>
struct tiled_src0_slot {
    typename tiled_fmt_tile<Fmt>::type * tile = nullptr;
    ~tiled_src0_slot() { delete tile; }
};
template <typename Fmt>
static tiled_src0_slot<Fmt> & tiled_get_src0_slot() {
    static thread_local tiled_src0_slot<Fmt> s;
    return s;
}

// process-wide switches, read once per process. The plan and the compute
// hook both go through the gate below, so one read keeps the wdata
// reservation and the take decision in sync.
//
// GGML_CPU_TILED_MM: master switch, on by default; the gate below then
// decides take/fallback per op. Setting it to 0 opts out of the feature
// entirely (no take, no reservation).
// GGML_CPU_TILED_MM_FORCE: test/bench only; take the tiled path even where
// the profitability check below rejects. Subordinate to the master switch
// and never relaxes the hard constraints.
static bool ggml_tiled_matmul_enabled(void) {
    static bool enabled = true;
    static bool inited  = false;
    if (!inited) {
        const char * env = getenv("GGML_CPU_TILED_MM");
        enabled = env == NULL || atoi(env) != 0;
        inited = true;
    }
    return enabled;
}

static bool ggml_tiled_matmul_forced(void) {
    static bool forced = false;
    static bool inited  = false;
    if (!inited) {
        const char * env = getenv("GGML_CPU_TILED_MM_FORCE");
        forced = env != NULL && atoi(env) == 1;
        inited = true;
    }
    return forced;
}

// shape/type gate; anything not supported here runs the old path
bool ggml_tiled_matmul_supported(const struct ggml_tensor * src0,
                                 const struct ggml_tensor * src1,
                                 const struct ggml_tensor * dst) {
    if (!ggml_tiled_matmul_enabled()) {
        return false;
    }

    // hard constraints: the kernel is only correct/defined for these; the
    // force switch never relaxes them
    // repack-buffer weights hold a repacked layout, not the raw quants this
    // path reads; the stock path selects the repack kernel via src0->extra
    if (src0->extra != NULL) {
        return false;
    }
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
    // src0 batch dims broadcast over src1 batch dims (the kernel maps src0
    // batch coords down with i02 = i12/r2, i03 = i13/r3, as in the stock path)
    if (src0->ne[2] == 0 || src0->ne[3] == 0 ||
        src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
        return false;
    }
    // src1 rows must be addressable as contiguous rows (wdata is contiguous by construction)
    if (src1->type == GGML_TYPE_Q8_K && !ggml_is_contiguous(src1)) {
        return false;
    }

    // the force switch (test/bench only) takes what the profitability
    // check below would reject
    if (ggml_tiled_matmul_forced()) {
        return true;
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
static void ggml_compute_forward_mul_mat_tiled_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const tiled_src1_interleave & src1_inter) {

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

    const int8_t * qv = src1_inter.qv;
    const int64_t nr1_pad = src1_inter.nr1_pad;

    tiled_src0_slot<Fmt> & slot = tiled_get_src0_slot<Fmt>();
    if (!slot.tile) {
        slot.tile = new typename tiled_fmt_tile<Fmt>::type();
    }
    if (!tiled_ws.src1) {
        tiled_ws.src1 = new tiled_tile_src1();
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

    const int64_t src0_stride = nb01 / src0_bs;  // blocks between src0 rows
    const int64_t src1_stride = (src1->type == vec_dot_type ? src1->nb[1] : row_size) / src1_bs;

    const int64_t TILE = 256;
    const int64_t MICRO = 16;

    // === logical 256x256 tiles (macrotiles) ===
    // The ir1 macrotile is additionally clamped at the src0 batch (ne11) boundary:
    // the tiles require a constant batch index (i12/i13) within one window. Advance
    // by the clamped end (not a fixed 256) so no rows are skipped at a batch boundary.
    for (int64_t tile_n1 = ir1_start; tile_n1 < ir1_end; ) {
        int64_t tile_n1_end = MIN(tile_n1 + TILE, ir1_end);
        const int64_t bnd = (tile_n1 / ne11 + 1) * ne11;
        if (bnd < tile_n1_end) {
            tile_n1_end = bnd;
        }

        const int n_src1 = (int) (tile_n1_end - tile_n1);

        // src0 batch coords (constant within the clamped window)
        const int64_t i13 = tile_n1 / (ne12 * ne11);
        const int64_t i12 = (tile_n1 - i13 * ne12 * ne11) / ne11;
        // within-batch row; dst_base already holds the i12/i13 batch offset, so the
        // row offset must use i11 (not the flattened tile_n1, which spans all batch dims)
        const int64_t i11 = tile_n1 - i13 * ne12 * ne11 - i12 * ne11;

        // dst batches == src1 batches (ggml_mul_mat), so the loop batch
        // coords are in src1 space; src0 batches are broadcast over them
        // (ne02|ne12, ne03|ne13), map down into src0's own batch for the src0 tile
        const int64_t r2 = ne12 / ne02;
        const int64_t r3 = ne13 / ne03;
        const int64_t i02 = i12 / r2;
        const int64_t i03 = i13 / r3;

        const char * src0_base = (const char *) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3];
        char * dst_base = (char *) dst->data + i12 * nb2 + i13 * nb3;

        const block_q8_K * src1_rows = (const block_q8_K *) ((const char *) wdata + tile_n1 * src1_stride * src1_bs);

        for (int64_t tile_n0 = ir0_start; tile_n0 < ir0_end; tile_n0 += TILE) {
            const int64_t tile_n0_end = MIN(tile_n0 + TILE, ir0_end);
            const int n_src0 = (int) (tile_n0_end - tile_n0);

            const char * src0_rows = src0_base + tile_n0 * nb01;

            float * c_curr = (float *) (dst_base + tile_n0 * nb0 + i11 * nb1);

            // j-major buffer: zeroed once per macrotile, accumulated over all slabs and
            // microtiles (each a cheap contiguous add), then transposed into dst once so
            // every dst element is written exactly once (a per-microtile strided dst RMW
            // measured ~30% of runtime)
            memset(tiled_ws.acc, 0, (size_t)TILED_TILE_ROWS * TILED_TILE_ROWS * sizeof(float));

            // Compute tiles of length 256 towards our NXN output block
            for (int64_t ib = 0; ib < ne00; ib += TILE) {
                const int kblk = (int) (ib / TILE);
                tiled_unpack_src0(Fmt(), src0_rows + kblk * src0_bs, src0_stride, n_src0, slot.tile);
                // tile_n1 is the flattened global row index of the window's first row
                tiled_unpack_src1_q8_K(src1_rows + kblk, src1_stride, n_src1, tiled_ws.src1,
                                       qv, nr1_pad, tile_n1, kblk);

                for (int64_t iir0 = tile_n0; iir0 < tile_n0_end; iir0 += MICRO) {
                    for (int64_t iir1 = tile_n1; iir1 < tile_n1_end; iir1 += MICRO) {
                        // 16x16 microtile window over the macrotile (tile-local coords);
                        // the kernel processes the full 16x16 unconditionally, rows/cols
                        // past the window edges hold harmless tile garbage (src1 rows are
                        // zero-padded at unpack) and the store below drops them
                        tiled_run_microtile(*slot.tile, *tiled_ws.src1,
                            (int) (iir0 - tile_n0), (int) (iir1 - tile_n1),
                            tiled_ws.acc, TILED_TILE_ROWS);
                    }
                }
            }

            tiled_store_window(tiled_ws.acc, n_src0, n_src1, TILED_TILE_ROWS, c_curr, ldc);
        }
        tile_n1 = tile_n1_end;
    }
}

template <typename Fmt>
static void ggml_compute_forward_mul_mat_tiled_fmt(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

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

    // interleave the whole tensor's src1 codes once; no-op on non-VNNI builds
    const tiled_src1_interleave src1_inter = tiled_prepare_src1_interleave(
        params, src1, vec_dot_type, ne10, ne11 * ne12 * ne13, ith, nth);

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

        ggml_compute_forward_mul_mat_tiled_one_chunk<Fmt>(params, dst, ir0_start, ir0_end, ir1_start, ir1_end,
                                                          src1_inter);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

bool ggml_compute_forward_mul_mat_tiled(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {
    // the stock path is the reference; --use-ref must stay on it
    if (params->use_ref) {
        return false;
    }
    if (!ggml_tiled_matmul_supported(dst->src[0], dst->src[1], dst)) {
        return false;
    }
    switch (dst->src[0]->type) {
        case GGML_TYPE_Q4_K:
            ggml_compute_forward_mul_mat_tiled_fmt<tiled_fmt_q4_K>(params, dst);
            break;
        case GGML_TYPE_Q5_K:
            ggml_compute_forward_mul_mat_tiled_fmt<tiled_fmt_q5_K>(params, dst);
            break;
        case GGML_TYPE_Q6_K:
            ggml_compute_forward_mul_mat_tiled_fmt<tiled_fmt_q6_K>(params, dst);
            break;
        case GGML_TYPE_Q3_K:
            ggml_compute_forward_mul_mat_tiled_fmt<tiled_fmt_q3_K>(params, dst);
            break;
        case GGML_TYPE_Q2_K:
            ggml_compute_forward_mul_mat_tiled_fmt<tiled_fmt_q2_K>(params, dst);
            break;
        default:
            return false;
    }
    return true;
}
