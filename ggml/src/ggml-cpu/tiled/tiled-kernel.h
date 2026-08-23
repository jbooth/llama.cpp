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

// B tile: activation side, built from q8_K (wdata), shared by all A formats.
// Rows are padded to a multiple of 16 (zeroed) so column microtiles are dense.
// Members are 64B aligned so the base (and every member, whose offsets are
// multiples of 64B) is cache-line aligned: the VNNI kernel reads qv and d with
// 64B vectors, and a 64B access at a non-64B offset straddles two cache lines.
struct tiled_tile_b {
    alignas(64) int8_t  q[TILED_TILE_ROWS * TILED_TILE_K];            // signed q8 codes, natural [row][k]
    alignas(64) int8_t  qv[(TILED_TILE_K / 4) * TILED_TILE_ROWS * 4]; // VNNI interleaved [k/4][row][4]
    alignas(64) int16_t bsums[(TILED_TILE_K / 16) * TILED_TILE_ROWS]; // s-major per-16 sums, [s][row]
    float       d[TILED_TILE_ROWS];
};

// A tile: weight side, one template instantiation per format.
template <int SUBBLK, bool HAS_MIN, int BIAS>
struct tiled_tile_a {
    static constexpr bool HAS_MIN_V = HAS_MIN; // re-exposed for the kernel templates (which take only the type)
    static constexpr int BIAS_V = BIAS;
    static constexpr int NB = TILED_TILE_K / SUBBLK; // subblocks per 256-K block
    alignas(32) uint8_t q[TILED_TILE_ROWS * TILED_TILE_K]; // unsigned codes
    float    d[TILED_TILE_ROWS];
    float    dmin[TILED_TILE_ROWS];         // used when HAS_MIN
    int8_t   sc[TILED_TILE_ROWS * NB];      // per-subblock scale (q3_K: stored as raw-32)
    int8_t   mn[TILED_TILE_ROWS * NB];      // per-subblock min, used when HAS_MIN
    // NB: a BIAS != 0 format needs NO A-side code sum. sum(c*qB) with c = u - BIAS
    // = sum(u*qB) - BIAS*sum(qB): the bias term is BIAS times the B-side per-subblock
    // bsum (tile_b->bsums), which the kernel already combines. No A sumq is stored.
};

// worst-case instantiation (SUBBLK = 16, HAS_MIN) + B tile must stay under the per-thread budget
static_assert(sizeof(tiled_tile_a<16, true, 0>) + sizeof(tiled_tile_b) < 512 * 1024,
              "tiled tile memory budget exceeded");

// q4_K and q5_K share the tile layout: 32-wide subblocks, min, no bias, 1-byte codes
typedef tiled_tile_a<32, true, 0> tiled_tile_a_q4_K;
typedef tiled_tile_a<32, true, 0> tiled_tile_a_q5_K;
// q6_K/q3_K/q2_K use 16-wide subblocks (NB=16). q6_K/q3_K carry a BIAS (corrected via
// the B bsums); q2_K uses per-16 min, no bias.
typedef tiled_tile_a<16, false, 32> tiled_tile_a_q6_K;
typedef tiled_tile_a<16, false, 4>  tiled_tile_a_q3_K;
typedef tiled_tile_a<16, true, 0>   tiled_tile_a_q2_K;

// format tags select the A unpacker in the templated driver
struct tiled_fmt_q4_K {};
struct tiled_fmt_q5_K {};
struct tiled_fmt_q6_K {};
struct tiled_fmt_q3_K {};
struct tiled_fmt_q2_K {};

// map a format tag to its A tile type (used by the templated driver/workspace)
template <typename Fmt> struct tiled_fmt_tile;
template <> struct tiled_fmt_tile<tiled_fmt_q4_K> { using type = tiled_tile_a_q4_K; };
template <> struct tiled_fmt_tile<tiled_fmt_q5_K> { using type = tiled_tile_a_q5_K; };
template <> struct tiled_fmt_tile<tiled_fmt_q6_K> { using type = tiled_tile_a_q6_K; };
template <> struct tiled_fmt_tile<tiled_fmt_q3_K> { using type = tiled_tile_a_q3_K; };
template <> struct tiled_fmt_tile<tiled_fmt_q2_K> { using type = tiled_tile_a_q2_K; };

// unpack one (window, k-block) into a tile: n_rows <= TILED_TILE_ROWS rows,
// row r at rows + r*row_stride (in blocks)
void tiled_unpack_a_q4_K(const block_q4_K * rows, int64_t row_stride, int n_rows, tiled_tile_a_q4_K * tile);
void tiled_unpack_a_q5_K(const block_q5_K * rows, int64_t row_stride, int n_rows, tiled_tile_a_q5_K * tile);
void tiled_unpack_a_q6_K(const block_q6_K * rows, int64_t row_stride, int n_rows, tiled_tile_a_q6_K * tile);
void tiled_unpack_a_q3_K(const block_q3_K * rows, int64_t row_stride, int n_rows, tiled_tile_a_q3_K * tile);
void tiled_unpack_a_q2_K(const block_q2_K * rows, int64_t row_stride, int n_rows, tiled_tile_a_q2_K * tile);
void tiled_unpack_b_q8_K(const block_q8_K * rows, int64_t row_stride, int n_rows, tiled_tile_b * tile);

// tag-dispatched unpack: rows is the base of one (window, k-block), cast to the format's block type
inline void tiled_unpack_a(tiled_fmt_q4_K, const void * rows, int64_t row_stride, int n_rows, tiled_tile_a_q4_K * tile) {
    tiled_unpack_a_q4_K((const block_q4_K *) rows, row_stride, n_rows, tile);
}
inline void tiled_unpack_a(tiled_fmt_q5_K, const void * rows, int64_t row_stride, int n_rows, tiled_tile_a_q5_K * tile) {
    tiled_unpack_a_q5_K((const block_q5_K *) rows, row_stride, n_rows, tile);
}
inline void tiled_unpack_a(tiled_fmt_q6_K, const void * rows, int64_t row_stride, int n_rows, tiled_tile_a_q6_K * tile) {
    tiled_unpack_a_q6_K((const block_q6_K *) rows, row_stride, n_rows, tile);
}
inline void tiled_unpack_a(tiled_fmt_q3_K, const void * rows, int64_t row_stride, int n_rows, tiled_tile_a_q3_K * tile) {
    tiled_unpack_a_q3_K((const block_q3_K *) rows, row_stride, n_rows, tile);
}
inline void tiled_unpack_a(tiled_fmt_q2_K, const void * rows, int64_t row_stride, int n_rows, tiled_tile_a_q2_K * tile) {
    tiled_unpack_a_q2_K((const block_q2_K *) rows, row_stride, n_rows, tile);
}

// Accumulate one 16x16 microtile (A rows [i0, i0+16), B cols [j0, j0+16))
// over the full 256-K slab held in the tiles into a j-major float buffer
// (row width buf_stride): buf[i*buf_stride + j] += partial. i0/j0 are
// multiples of TILED_MICRO and within TILED_TILE_ROWS; rows/cols past the
// window hold harmless tile garbage and the driver's store drops them.
// No store to C here: the driver holds the buffer across the 256-K slabs
// and transpose-stores it once, so each C element is written a single time.
template <typename T> // T = tiled_tile_a<...>
void tiled_run_microtile(const T & a, const tiled_tile_b & b,
                         int i0, int j0, float * buf, int buf_stride);

// Transpose-store the j-major buffer to C: C[i][j] = c[i + j*ldc] (i contiguous),
// n_a rows x n_b cols, buffer row width buf_stride. Written with = (the buffer holds
// the full sum for the window and the region is thread-exclusive).
void tiled_store_window(const float * buf, int n_a, int n_b, int buf_stride, float * c, size_t ldc);