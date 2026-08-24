#pragma once

// Tiled matmul kernel: tile structs, unpackers, microtile kernel.
// C++ for the per-format template config (SUBBLK, HAS_MIN, BIAS); C-style code otherwise.
// K-quant weights x q8_K activations; per-ISA branches in the kernel (scalar,
// AVX, AVX2, AVX512-VNNI).

#include "ggml-quants.h"

#include <stddef.h>
#include <stdint.h>

#define TILED_TILE_K    256 // one QK_K block
#define TILED_TILE_ROWS 256 // max window rows, ragged at edges
#define TILED_MICRO     16  // microtile edge

// src1 tile: activation side, built from q8_K (wdata), shared by all src0 formats.
// Rows are padded to a multiple of 16 (zeroed) so column microtiles are dense.
// q and qv alias the same 64KB and are never both active: the scalar/AVX2
// bodies read q ([row][k]), the VNNI body reads qv ([k/4][row][4], 16 rows x
// 4B contiguous per k-group, the dpbusd 16-lane operand). The unpacker
// builds whichever one its own build's kernel consumes.
// Members are 64B aligned so the base (and every member, whose offsets are
// multiples of 64B) is cache-line aligned: the VNNI kernel reads qv and d with
// 64B vectors, and a 64B access at a non-64B offset straddles two cache lines.
struct tiled_tile_src1 {
    union {
        alignas(64) int8_t  q[TILED_TILE_ROWS * TILED_TILE_K];            // signed q8 codes, natural [row][k]
        alignas(64) int8_t  qv[(TILED_TILE_K / 4) * TILED_TILE_ROWS * 4]; // VNNI interleaved [k/4][row][4]
    };
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
    // subblock dot:  out += d * sc[r][s] * dot(codes)  -  dmin * mn[r][s] * sum(src1 codes)
    // sc scales the int code dot, mn scales the src1-side code sum (tile->bsums); mn is
    // only used by HAS_MIN formats (q2/q4/q5_K). q3_K stores sc as (raw 6-bit scale - 32)
    // (its q1 code offset is handled via BIAS). q6_K uses only sc.
    // int32 though the values fit int8: broadcast straight from memory, no per-use sign-extend (12.9)
    int32_t   sc[TILED_TILE_ROWS * NB];      // per-subblock scale (q3_K: stored as raw-32)
    int32_t   mn[TILED_TILE_ROWS * NB];      // per-subblock min, used when HAS_MIN
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

// unpack one (window, k-block) into a tile: n_rows <= TILED_TILE_ROWS rows,
// row r at rows + r*row_stride (in blocks)
void tiled_unpack_src0_q4_K(const block_q4_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q4_K * tile);
void tiled_unpack_src0_q5_K(const block_q5_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q5_K * tile);
void tiled_unpack_src0_q6_K(const block_q6_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q6_K * tile);
void tiled_unpack_src0_q3_K(const block_q3_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q3_K * tile);
void tiled_unpack_src0_q2_K(const block_q2_K * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q2_K * tile);
// src1 unpack: n_rows <= TILED_TILE_ROWS rows, row r at rows + r*row_stride (in blocks).
// d and bsums always come from the natural q8_K rows; the VNNI build takes the
// codes from the one-shot interleaved region (qv_glob) when provided, else builds
// them per call (r_glob = global index of the window's first row, kblk = slab)
void tiled_unpack_src1_q8_K(const block_q8_K * rows, int64_t row_stride, int n_rows, tiled_tile_src1 * tile,
                            const int8_t * qv_glob, int64_t n_rows_pad, int64_t r_glob, int64_t kblk);

// One-shot, all threads: scatter the whole src1 tensor's q8 codes into the flat
// [slab][k/4-in-slab][row][4] region (one code byte per element, rows padded to
// 16 and zeroed past n_rows). d and bsums stay in the natural layout. Each call
// covers rows [r_start, r_end) (16-aligned, may reach n_rows_pad), so callers
// split the tensor into row groups per thread
void tiled_interleave_src1_q8_K(const block_q8_K * rows, int64_t row_stride,
                                int64_t r_start, int64_t r_end,
                                int64_t n_k, int64_t n_rows, int64_t n_rows_pad, int8_t * qv_glob);

// tag-dispatched unpack: rows is the base of one (window, k-block), cast to the format's block type
inline void tiled_unpack_src0(tiled_fmt_q4_K, const void * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q4_K * tile) {
    tiled_unpack_src0_q4_K((const block_q4_K *) rows, row_stride, n_rows, tile);
}
inline void tiled_unpack_src0(tiled_fmt_q5_K, const void * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q5_K * tile) {
    tiled_unpack_src0_q5_K((const block_q5_K *) rows, row_stride, n_rows, tile);
}
inline void tiled_unpack_src0(tiled_fmt_q6_K, const void * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q6_K * tile) {
    tiled_unpack_src0_q6_K((const block_q6_K *) rows, row_stride, n_rows, tile);
}
inline void tiled_unpack_src0(tiled_fmt_q3_K, const void * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q3_K * tile) {
    tiled_unpack_src0_q3_K((const block_q3_K *) rows, row_stride, n_rows, tile);
}
inline void tiled_unpack_src0(tiled_fmt_q2_K, const void * rows, int64_t row_stride, int n_rows, tiled_tile_src0_q2_K * tile) {
    tiled_unpack_src0_q2_K((const block_q2_K *) rows, row_stride, n_rows, tile);
}

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

// Transpose-store the j-major buffer to dst: dst[i][j] = c[i + j*ldc] (i contiguous),
// n_src0 rows x n_src1 cols, buffer row width buf_stride. Written with = (the buffer holds
// the full sum for the window and the region is thread-exclusive).
void tiled_store_window(const float * buf, int n_src0, int n_src1, int buf_stride, float * c, size_t ldc);