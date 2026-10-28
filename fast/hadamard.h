/* LiGGUF - PrismML Hadamard folding (prism.hadamard.*)
 *
 * Some ternary checkpoints store weights that have been rotated into the
 * Hadamard basis on the input axis. The matmul then becomes
 *
 *     out = W . (signs * (H_n x))
 *
 * where H_n is the normalized Sylvester-Walsh-Hadamard matrix of order n
 * (block_size), applied blockwise along the input dimension, and `signs` is an
 * optional elementwise +/-1 vector.
 *
 * Reference: ggml-org/llama.cpp (upstream) and PrismML-Eng/llama.cpp branch
 * `prism` — src/llama-model.cpp (rotation matrix construction) and
 * src/llama-impl.h (llama_mul_mat_hadamard).
 *
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#pragma once

#include "lil_gguf.h"

/* Build the [n*n] normalized Sylvester-Walsh-Hadamard matrix into `rot`.
 * rot[row*n + col] = (popcount(row & col) odd ? -1 : +1) / sqrt(n) */
void hadamard_build_matrix(ftensor& rot, int n);

/* Forward transform: rotate `x` in place along blocks of `block_size`.
 * `signs` (when non-empty and of width block_size) is applied first. */
void hadamard_apply_forward(ftensor& x, const ftensor& rot, int block_size,
                            const ftensor* signs);

/* Inverse transform used after a row lookup (tied output embeddings). The
 * normalized Hadamard matrix is its own inverse up to scaling, so the same
 * routine applies; we expose it separately for clarity at the call site. */
void hadamard_apply_inverse(ftensor& x, const ftensor& rot, int block_size,
                            const ftensor* signs);
