#pragma once

// Tiled matmul kernel API: tile structs, format tags, per-ISA microtile kernel
// entry points, and the arch-specific src1 code unpack op (VNNI).
// C++ for the per-format template config (SUBBLK, HAS_MIN, BIAS); C-style code otherwise.
// K-quant weights x q8_K activations; per-ISA branches in the kernel (scalar,
// AVX, AVX2, AVX512-VNNI).

#include "ggml-quants.h"

#include <stddef.h>
#include <stdint.h>

// Defined when this arch's kernel reads the src1 tile codes in a non-natural
// order (x86 VNNI: [k/4][row][4], fed from the up-front interleaved region).
// The kernel then provides tiled_unpack_src1_q8_K_kernel, which the driver's
// src1 unpack calls instead of its generic per-row memcpy
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
#define KERNEL_SRC1_UNPACK 1
#endif

#define TILED_TILE_K    256 // one QK_K block
#define TILED_TILE_ROWS 256 // max window rows, ragged at edges
#define TILED_MICRO     16  // microtile edge (also the bsums code-sum granularity)

// src1 tile: built from q8_K (wdata), shared by all src0 formats.
struct tiled_tile_src1 {
    // signed q8 codes, 256 x 256, 64B aligned. The compiled kernel -- not the
    // unpack -- chooses the byte order: natural [row][k] on the scalar/AVX/AVX2
    // tiers, [k/4][row][4] on VNNI (one 64B dpbusd vector per k-group x 16 rows).
    alignas(64) int8_t  q[TILED_TILE_ROWS * TILED_TILE_K];
    // s-major per-16 code sums, [s][row]; int32 (model stores int16): one 64B vector load
    // per (s, j0) on VNNI, no per-use cvt on the AVX tiers; L2-resident, width costs nothing
    alignas(64) int32_t bsums[(TILED_TILE_K / 16) * TILED_TILE_ROWS];
    // f32 (not f16): the model stores fp16, the unpack converts once
    float       d[TILED_TILE_ROWS];
};

// src0 tile: weight side, one template instantiation per format.
template <int SUBBLK, bool HAS_MIN, int BIAS>
struct tiled_tile_src0 {
    static constexpr bool HAS_MIN_V = HAS_MIN; // re-exposed for the kernel templates (which take only the type)
    static constexpr int BIAS_V = BIAS;
    static constexpr int NB = TILED_TILE_K / SUBBLK; // subblocks per 256-K block
    // one byte per element (4/5/6-bit value in the low bits); unsigned: maddubs/dpbusd take (u8, s8)
    alignas(32) uint8_t q[TILED_TILE_ROWS * TILED_TILE_K]; // unsigned codes
    // f32 (not f16): the model stores fp16, the unpack converts once
    float    d[TILED_TILE_ROWS];
    float    dmin[TILED_TILE_ROWS];         // used when HAS_MIN
    // per-subblock side coefficients, one (row, s) pair each, applied per exact integer
    // subblock dot:  out += d * scales[r][s] * dot(codes)  -  dmin * mins[r][s] * sum(src1 codes)
    // scales (the model block's scales field) scales the int code dot, mins scales the src1-side
    // code sum (tile->bsums); mins is only used by HAS_MIN formats (q2/q4/q5_K). q3_K stores
    // scales as (raw 6-bit scale - 32) (its q1 code offset is handled via BIAS). q6_K uses
    // only scales.
    // int32 though the values fit int8: broadcast straight from memory, no per-use sign-extend (12.9)
    int32_t   scales[TILED_TILE_ROWS * NB];  // per-subblock scale (q3_K: stored as raw-32)
    int32_t   mins[TILED_TILE_ROWS * NB];      // per-subblock min, used when HAS_MIN
    // NB: a BIAS != 0 format needs NO src0-side code sum. sum(src1 codes) with c = u - BIAS
    // = sum(u*src1 code) - BIAS*sum(src1 code): the bias term is BIAS times the src1-side per-subblock
    // bsum (tile_src1->bsums), which the kernel already combines. No src0 sumq is stored.
};

// worst-case instantiation (SUBBLK = 16, HAS_MIN) + src1 tile must stay under the per-thread budget
static_assert(sizeof(tiled_tile_src0<16, true, 0>) + sizeof(tiled_tile_src1) < 512 * 1024,
              "tiled tile memory budget exceeded");

// q4_K and q5_K share the tile layout: 32-wide subblocks, min, no bias, 1-byte codes
typedef tiled_tile_src0<32, true, 0> tiled_tile_src0_q4_K;
typedef tiled_tile_src0<32, true, 0> tiled_tile_src0_q5_K;
// q6_K/q3_K/q2_K use 16-wide subblocks (NB=16). q6_K/q3_K carry a BIAS (corrected via
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

// VNNI-only (KERNEL_SRC1_UNPACK): the dpbusd kernel reads the src1 tile codes
// in the [k/4][row][4] order, so the driver's generic per-row memcpy cannot
// fill the tile; this op does, from the up-front interleaved region (qv,
// non-null; r_start = global index of the window's first row, kblk = slab).
// d and bsums are still filled by the driver's unpack
#if defined(KERNEL_SRC1_UNPACK)
void tiled_unpack_src1_q8_K_kernel(int n_rows, tiled_tile_src1 * tile,
                                   const int8_t * qv, int64_t nr1_pad, int64_t r_start, int64_t kblk);
#endif

// info for the src1 interleave region built by tiled_prepare_src1_interleave:
// qv = base of the up-front [slab][k/4][row][4] code region (null when not
// built, i.e. non-VNNI), nr1_pad = the region's row count padded to 16 (0 then)
struct tiled_src1_interleave {
    const int8_t * qv;
    int64_t       nr1_pad;
};

// one-shot, all-threads build of the src1 interleave region: partitions the
// tiled_interleave_src1_q8_K scatter across threads, sizes/locates the wdata
// region, and barriers the F32->q8 conversion first. No-op on non-VNNI (returns
// {nullptr, 0}); the driver always calls it
struct ggml_compute_params;
tiled_src1_interleave tiled_prepare_src1_interleave(const struct ggml_compute_params * params,
                                                    const struct ggml_tensor * src1,
                                                    enum ggml_type vec_dot_type,
                                                    int64_t ne10, int64_t nr1, int ith, int nth);

// Accumulate one 16x16 microtile (src0 rows [i0, i0+16), src1 cols [j0, j0+16))
// over the full 256-K slab held in the tiles into a j-major float buffer
// (row width buf_stride): buf[i*buf_stride + j] += partial. i0/j0 are
// multiples of TILED_MICRO and within TILED_TILE_ROWS; rows/cols past the
// window hold harmless tile garbage and the driver's store drops them.
// No store to dst here: the driver holds the buffer across the 256-K slabs
// and transpose-stores it once, so each dst element is written a single time.
template <typename T> // T = tiled_tile_src0<...>
void tiled_run_microtile(const T & src0, const tiled_tile_src1 & src1,
                         int i0, int j0, float * buf, int buf_stride);

// Transpose-store the j-major buffer to dst: dst[i][j] = dst[i + j*dst_stride] (i contiguous),
// n_src0 rows x n_src1 cols, buffer row width buf_stride. Written with = (the buffer holds
// the full sum for the window and the region is thread-exclusive).
void tiled_store_window(const float * buf, int n_src0, int n_src1, int buf_stride, float * dst, size_t dst_stride);