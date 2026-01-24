#pragma once


#ifdef __cplusplus
extern "C" {
#endif

void ggml_compute_forward_mul_mat_tiled_implicit(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst);
#ifdef __cplusplus
}
#endif