#pragma once


#ifdef __cplusplus
extern "C" {
#endif

// Extra wdata reservation size required, if any (see VNNI impl)
size_t ggml_tiled_extra_wdata_len(int64_t ne10, int64_t nr1);

// tiled K-quant matmul; returns true if the op was computed here, 
// false to fall through to the stock path
bool ggml_compute_forward_mul_mat_tiled(const struct ggml_compute_params * params,
                                        struct ggml_tensor * dst);

// mul_mat_id (MoE expert FFN) variant of the tiled path. Same entry/driver/
// one_chunk ladder as the dense one, called from ggml_compute_forward_mul_mat_id
// after the stock grouping pass: matrix_rows holds the stock row grouping as
// (slot, token) int32 pairs, expert-major (n_as * ids->ne[0] * ids->ne[1] pairs),
// scratch is the per-thread region base (see ggml_tiled_mul_mat_id_extra_wdata_len).
bool ggml_compute_forward_mul_mat_id_tiled(const struct ggml_compute_params * params,
                                           struct ggml_tensor * dst,
                                           const int32_t * matrix_rows,
                                           const int64_t * matrix_row_counts,
                                           char * scratch);

// Extra wdata reservation size for the mul_mat_id tiled path, if any:
// per-thread gathered q8_K rows (256) + interleave region (VNNI builds only)
size_t ggml_tiled_mul_mat_id_extra_wdata_len(int64_t ne10, int64_t n_tasks);

// Per-expert gate of the mul_mat_id tiled path (cne1 >= 64, forced bypass).
// The hook in ggml-cpu.c applies the same check to skip the stock experts.
bool ggml_tiled_mul_mat_id_expert_supported(int64_t cne1);
#ifdef __cplusplus
}
#endif
