#pragma once


#ifdef __cplusplus
extern "C" {
#endif

// shape/type gate for the tiled K-quant matmul path; false means run the
// stock path. Includes the env switches (see tiled.cpp): GGML_CPU_TILED_MM
// master switch (default on, 0 opts out entirely) and GGML_CPU_TILED_MM_FORCE
// (test/bench only; takes shapes the profitability check rejects, never
// relaxes the hard constraints), so ggml_graph_plan and the compute hook
// share one predicate (the plan reserves the VNNI interleave region only
// when the take is possible)
bool ggml_tiled_matmul_supported(const struct ggml_tensor * src0,
                                 const struct ggml_tensor * src1,
                                 const struct ggml_tensor * dst);

// tiled K-quant matmul; returns true if the op was computed here (caller
// returns), false to fall through to the stock path
bool ggml_compute_forward_mul_mat_tiled(const struct ggml_compute_params * params,
                                        struct ggml_tensor * dst);
#ifdef __cplusplus
}
#endif