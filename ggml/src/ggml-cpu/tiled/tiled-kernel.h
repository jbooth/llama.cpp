#pragma once

// Tiled matmul kernel API: tile structs, kernel definitions

// Currently only optimized for x86, new architectures should implement:
// tiled_run_microtile:  16x16 microkernel
// tiled_store_window: Writeback of 256x256 window (default fallback may be good enough)
// bit unpacking routines: tiled_unpk_nib4, tiled_unpk_2bit, tiled_unpk_or
#include "ggml-quants.h"
#include "ggml.h"
#include "ggml-cpu-impl.h" // ggml_compute_params; no-op for the consumers, which include it first

#include <stddef.h>
#include <stdint.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define TILED_TILE_K    256 // one QK_K block
#define TILED_TILE_ROWS 256 // max window rows, ragged at edges
#define TILED_MICRO     16  // microtile edge (also the bsums code-sum granularity)

// src1 tile: built from q8_K (wdata), shared by all src0 formats.
struct tiled_tile_src1 {
    // q8 codes, one byte per element. Note for VNNI these are reshaped + transposed to be suitable for dpbusd.
    alignas(64) int8_t  q[TILED_TILE_ROWS * TILED_TILE_K];
    // per-16 code sums from q8_k (int16), widened to int32 so the kernels load them directly, no per-use cvt
    alignas(64) int32_t bsums[(TILED_TILE_K / 16) * TILED_TILE_ROWS];
    // f32 (not f16): q8_k stores fp16, the unpack converts once
    float       d[TILED_TILE_ROWS];
};

// src0 tile: weight side, one template instantiation per format.
template <int SUBBLK, bool HAS_MIN, int BIAS>
struct tiled_tile_src0 {
    static constexpr bool HAS_MIN_V = HAS_MIN; // re-exposed for the kernel templates (which take only the type)
    static constexpr int BIAS_V = BIAS;
    static constexpr int NB = TILED_TILE_K / SUBBLK; // subblocks per 256-K block
    
    // unsigned quants, expanded to 8bit
    alignas(32) uint8_t q[TILED_TILE_ROWS * TILED_TILE_K]; 
    // f32 (not f16): the model stores fp16, the unpack converts once
    float    d[TILED_TILE_ROWS];  // One d from each input block, widened to f32
    float    dmin[TILED_TILE_ROWS]; // dmin from each input block (if applicable), widened to F32
    // per-subblock side coefficients
    int32_t   scales[TILED_TILE_ROWS * NB];  // per-subblock scale, stored as int32_t
    int32_t   mins[TILED_TILE_ROWS * NB];      // per-subblock min, used when HAS_MIN
};

// Ensure total size under 512kb for L2 cache fit
static_assert(sizeof(tiled_tile_src0<16, true, 0>) + sizeof(tiled_tile_src1) < 512 * 1024,
              "tiled tile memory budget exceeded");

// q4_K and q5_K share the tile layout: 32-wide subblocks, min, no bias, 1-byte codes
typedef tiled_tile_src0<32, true, 0> tiled_tile_src0_q4_K;
typedef tiled_tile_src0<32, true, 0> tiled_tile_src0_q5_K;
// q6_K/q3_K/q2_K use 16-wide subblocks (NB=16). q6_K/q3_K carry a bias (corrected via
// the src1 bsums); q2_K uses per-16 min, no bias.
typedef tiled_tile_src0<16, false, 32> tiled_tile_src0_q6_K;
typedef tiled_tile_src0<16, false, 4>  tiled_tile_src0_q3_K;
typedef tiled_tile_src0<16, true, 0>   tiled_tile_src0_q2_K;

// format tags select the src0 unpacker in the templated driver
struct tiled_fmt_q4_K {};
struct tiled_fmt_q5_K {};
struct tiled_fmt_q6_K {};
struct tiled_fmt_q3_K {};
struct tiled_fmt_q2_K {};

// map a format tag to its src0 tile type (used by the templated driver/workspace)
template <typename Fmt> struct tiled_fmt_tile;
template <> struct tiled_fmt_tile<tiled_fmt_q4_K> { using type = tiled_tile_src0_q4_K; };
template <> struct tiled_fmt_tile<tiled_fmt_q5_K> { using type = tiled_tile_src0_q5_K; };
template <> struct tiled_fmt_tile<tiled_fmt_q6_K> { using type = tiled_tile_src0_q6_K; };
template <> struct tiled_fmt_tile<tiled_fmt_q3_K> { using type = tiled_tile_src0_q3_K; };
template <> struct tiled_fmt_tile<tiled_fmt_q2_K> { using type = tiled_tile_src0_q2_K; };


// 32-byte-unit unpack primitives for the src0 code expansion (the driver's
// per-format unpackers use them). All codes are bitfields with no overlap,
// so merging a flag into an extracted code uses OR (== ADD here, matches the
// reference kernels). ymm under __AVX2__, plain loops otherwise; the output
// bytes are bit-identical either way
#if defined(__AVX2__)
// packed 4-bit codes -> low nibbles (lo) + high nibbles (hi)
inline void tiled_unpk_nib4(const uint8_t * src, uint8_t * lo, uint8_t * hi) {
    const __m256i v = _mm256_loadu_si256((const __m256i *) src);
    // mask before the lane shift so bits do not cross byte boundaries
    _mm256_storeu_si256((__m256i *) lo, _mm256_and_si256(v, _mm256_set1_epi8(0x0F)));
    _mm256_storeu_si256((__m256i *) hi, _mm256_srli_epi32(_mm256_and_si256(v, _mm256_set1_epi8((int8_t) 0xF0)), 4));
}
// 2-bit values at bit offset S
template <int S> inline void tiled_unpk_2bit(const uint8_t * src, uint8_t * dst) {
    _mm256_storeu_si256((__m256i *) dst, _mm256_and_si256(
        _mm256_srli_epi32(_mm256_loadu_si256((const __m256i *) src), S), _mm256_set1_epi8(0x03)));
}
// OR the M-bit value at bit offset S of src into bit offset D of dst
template <int S, int D, int M>
inline void tiled_unpk_or(uint8_t * dst, const uint8_t * src) {
    const __m256i v = _mm256_slli_epi32(_mm256_and_si256(
        _mm256_srli_epi32(_mm256_loadu_si256((const __m256i *) src), S), _mm256_set1_epi8((uint8_t) M)), D);
    _mm256_storeu_si256((__m256i *) dst, _mm256_or_si256(_mm256_loadu_si256((const __m256i *) dst), v));
}
#else
inline void tiled_unpk_nib4(const uint8_t * src, uint8_t * lo, uint8_t * hi) {
    for (int l = 0; l < 32; l++) { lo[l] = (uint8_t) (src[l] & 0xF); hi[l] = (uint8_t) (src[l] >> 4); }
}
template <int S>
inline void tiled_unpk_2bit(const uint8_t * src, uint8_t * dst) {
    for (int l = 0; l < 32; l++) { dst[l] = (uint8_t) ((src[l] >> S) & 3); }
}
template <int S, int D, int M>
inline void tiled_unpk_or(uint8_t * dst, const uint8_t * src) {
    for (int l = 0; l < 32; l++) { dst[l] = (uint8_t) (dst[l] | (((src[l] >> S) & M) << D)); }
}
#endif


// Accumulate one 16x16 microtile (src0 rows [i0, i0+16), src1 cols [j0, j0+16))
// over the full 256-K slab held in the tiles into a j-major float buffer
// (row width buf_stride): buf[i*buf_stride + j] += partial.
// 
// No store to dst here: the driver holds the buffer across the 256-K slabs
// and transpose-stores it once, so each dst element is written a single time.
template <typename T> // T = tiled_tile_src0<...>
void tiled_run_microtile(const T & src0, const tiled_tile_src1 & src1,
                         int i0, int j0, float * buf, int buf_stride);

// Transpose-store the window buffer (256x256 max size) to dst: dst[(ri + t) + (rj + u) * dst_stride] = buf[(ri + t) * buf_stride + (rj + u)]
void tiled_store_window(const float * buf, int n_src0, int n_src1, int buf_stride, float * dst, size_t dst_stride);


// Defined when this arch's kernel reads the src1 tile codes in a non-natural
// order, in this case driver should call kernel methods `tiled_prepare_src1_interleave`
// and `tiled_unpack_src1_q8_K_kernel` to prepare the tensor and macrotiles, respectively.
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
#define KERNEL_SRC1_UNPACK 1
#endif


#if defined(KERNEL_SRC1_UNPACK)
// VNNI-only (KERNEL_SRC1_UNPACK): the dpbusd kernel reads the src1 tile codes
// in the [k/4][row][4] order instead of standard row-major
void tiled_unpack_src1_q8_K_kernel(int n_rows, tiled_tile_src1 * tile,
                                   const int8_t * qv, int64_t nr1_pad, int64_t r_start, int64_t kblk);

// geometry of the src1 interleave region in wdata (stubs on non-VNNI builds)
struct tiled_interleave_geom {
    int8_t       * qv;      // base of the [slab][k/4][row][4] code region (null when not built)
    int64_t       nr1_pad;  // row count padded to 16 (0 then)
    size_t        bytes;    // wdata reservation size for the region (0 then)
};

tiled_interleave_geom tiled_get_interleave_geom(const struct ggml_compute_params * params,
                                                const struct ggml_tensor * src1,
                                                enum ggml_type vec_dot_type,
                                                int64_t ne10, int64_t nr1);

// The q8 codes of the whole tensor are scattered into a flat [slab][k/4-in-slab][row][4] wdata region 
// (rows padded to 16, zeroed tail) so the per (window, slab) unpack becomes a contiguous copy.
void tiled_prepare_src1_interleave(const struct ggml_compute_params * params,
                                   const struct ggml_tensor * src1,
                                   enum ggml_type vec_dot_type,
                                   int64_t ne10, int64_t nr1, int ith, int nth);

#endif

