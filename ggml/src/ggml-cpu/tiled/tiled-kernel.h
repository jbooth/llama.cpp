#pragma once

// Tiled matmul kernel: tile structs, unpackers, microtile kernel.
// C++ for the per-format template config (SUBBLK, HAS_MIN, BIAS); C-style code otherwise.
// Phase 1: q5_K weights x q8_K activations, scalar kernel.

#include "ggml-quants.h"

#include <stddef.h>
#include <stdint.h>

#define TILED_TILE_K    256 // one QK_K block
#define TILED_TILE_ROWS 256 // max window rows, ragged at edges
#define TILED_MICRO     16  // microtile edge

// B tile: activation side, built from q8_K (wdata), shared by all A formats.
// Rows are padded to a multiple of 16 (zeroed) so column microtiles are dense.
struct tiled_tile_b {
    alignas(32) int8_t  q[TILED_TILE_ROWS * TILED_TILE_K];            // signed q8 codes, natural [row][k]
    alignas(32) int8_t  qv[(TILED_TILE_K / 4) * TILED_TILE_ROWS * 4]; // VNNI interleaved [k/4][row][4]
    alignas(16) int16_t bsums[TILED_TILE_ROWS * (TILED_TILE_K / 16)]; // per-16 sums of q
    float       d[TILED_TILE_ROWS];
};

// A tile: weight side, one template instantiation per format.
template <int SUBBLK, bool HAS_MIN, int BIAS>
struct tiled_tile_a {
    static constexpr int NB = TILED_TILE_K / SUBBLK; // subblocks per 256-K block
    alignas(32) uint8_t q[TILED_TILE_ROWS * TILED_TILE_K]; // unsigned codes
    float    d[TILED_TILE_ROWS];
    float    dmin[TILED_TILE_ROWS];         // used when HAS_MIN
    int8_t   sc[TILED_TILE_ROWS * NB];      // per-subblock scale
    int8_t   mn[TILED_TILE_ROWS * NB];      // per-subblock min, used when HAS_MIN
    int16_t  sumq[TILED_TILE_ROWS * NB];    // A bsums, used when BIAS != 0
};

// worst-case instantiation (SUBBLK = 16, HAS_MIN) + B tile must stay under the per-thread budget
static_assert(sizeof(tiled_tile_a<16, true, 0>) + sizeof(tiled_tile_b) < 512 * 1024,
              "tiled tile memory budget exceeded");

// phase 1 config: q5_K (32-wide subblocks, min, no bias)
typedef tiled_tile_a<32, true, 0> tiled_tile_a_q5_K;

// unpack one (window, k-block) into a tile: n_rows <= TILED_TILE_ROWS rows,
// row r at rows + r*row_stride (in blocks)
void tiled_unpack_a_q5_K(const block_q5_K * rows, int64_t row_stride, int n_rows, tiled_tile_a_q5_K * tile);
void tiled_unpack_b_q8_K(const block_q8_K * rows, int64_t row_stride, int n_rows, tiled_tile_b * tile);

// Accumulate one (window, k-block) partial sum into a j-major float buffer
// (row width buf_stride, j contiguous): buf[i*buf_stride + j] += partial, i < n_a, j < n_b.
// No store to C here: the driver holds the buffer across the 256-K blocks and
// transpose-stores it once, so each C element is written a single time.
void tiled_run_window_q5_K(const tiled_tile_a_q5_K & a, const tiled_tile_b & b,
                           int n_a, int n_b, float * buf, int buf_stride);

// Transpose-store the j-major buffer to C: C[i][j] = c[i + j*ldc] (i contiguous),
// n_a rows x n_b cols, buffer row width buf_stride. Written with = (the buffer holds
// the full sum for the window and the region is thread-exclusive).
void tiled_store_window(const float * buf, int n_a, int n_b, int buf_stride, float * c, size_t ldc);