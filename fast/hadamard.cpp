/* LiGGUF - PrismML Hadamard folding (prism.hadamard.*)
 * See hadamard.h for the contract and the upstream reference.
 *
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#include "hadamard.h"
#include "breadcrumb.h"
#include "common.h"

#include <string.h>

using namespace std;

static inline uint32_t hadamard_parity(uint32_t v)
{
    v ^= v >> 16;
    v ^= v >> 8;
    v ^= v >> 4;
    v ^= v >> 2;
    v ^= v >> 1;
    return v & 1u;
}

void hadamard_build_matrix(ftensor& rot, int n)
{
    rot.resize((size_t)n * n);
    const float scale = 1.0f / sqrtf((float)n);
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col)
            rot[(size_t)row * n + col] = hadamard_parity((uint32_t)row & (uint32_t)col) ? -scale : scale;
}

/* out[b*n + i] = sum_j rot[i*n + j] * in[b*n + j]  for each block b */
static void hadamard_blocks(const float* in, float* out, const float* rot, int n, int nblocks)
{
    for (int b = 0; b < nblocks; b++) {
        const float* src = in + (size_t)b * n;
        float* dst = out + (size_t)b * n;
        for (int i = 0; i < n; i++) {
            const float* row = rot + (size_t)i * n;
            float acc = 0.0f;
            for (int j = 0; j < n; j++) acc += row[j] * src[j];
            dst[i] = acc;
        }
    }
}

void hadamard_apply_forward(ftensor& x, const ftensor& rot, int block_size,
                            const ftensor* signs)
{
    const int n = block_size;
    if ((int)x.size() % n) {
        ERR("hadamard: activation width %zu is not a multiple of block size %d",x.size(),n);
        abort();
    }
    const int nblocks = (int)x.size() / n;
    if (!signs || (int)signs->size() != n) {
        // fast path: no sign vector, rotate in place through a scratch buffer
        ftensor tmp(x.size());
        hadamard_blocks(x.data(),tmp.data(),rot.data(),n,nblocks);
        memcpy(x.data(),tmp.data(),x.size() * sizeof(float));
        return;
    }

    // apply the sign flip first, then rotate
    ftensor tmp(x.size());
    for (int b = 0; b < nblocks; b++)
        for (int i = 0; i < n; i++)
            tmp[(size_t)b * n + i] = x[(size_t)b * n + i] * (*signs)[i];
    hadamard_blocks(tmp.data(),x.data(),rot.data(),n,nblocks);
}

void hadamard_apply_inverse(ftensor& x, const ftensor& rot, int block_size,
                            const ftensor* signs)
{
    // The normalized Hadamard matrix satisfies H = H^T and H*H = I, so the
    // inverse is the same transform. Signs commute with it for the diagonal
    // sign table, so the forward path is reused verbatim.
    hadamard_apply_forward(x,rot,block_size,signs);
}
