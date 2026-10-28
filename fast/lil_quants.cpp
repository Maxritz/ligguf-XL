/* LiGGUF - additional quantization codecs (dequant + native dot products)
 * Reference implementations follow ggml/src/ggml-quants.c and ggml-common.h
 * from ggml-org/llama.cpp, plus the PrismML fork for PQ2_0 / PTQ1_0.
 *
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#include "lil_quants.h"
#include "lil_math.h"
#include "breadcrumb.h"

#include <assert.h>
#include <string.h>

using namespace std;

const int8_t kvalues_mxfp4_tbl[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
const int8_t kvalues_iq4nl_tbl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

float ggml_e8m0_to_fp32(uint8_t e)
{
    // E8M0 has no mantissa and no sign; the exponent bias is 127.
    // ggml uses a "half" variant that keeps zero representable.
    if (e == 0) return 0.0f;
    uint32_t bits = (uint32_t)e << 23;
    float r;
    memcpy(&r,&bits,4);
    return r;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Dequantization — mirrors ggml-quants.c dequantize_row_* exactly.
 * ═══════════════════════════════════════════════════════════════════════════ */

void dequantize_row_q4_0(const block_q4_0* x, float* y, int64_t k)
{
    const int qk = QK4_0;
    const int64_t nb = k / qk;

    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;
            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void dequantize_row_q4_1(const block_q4_1* x, float* y, int64_t k)
{
    const int qk = QK4_1;
    const int64_t nb = k / qk;

    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const float m = fp16_to_fp32(x[i].m);
        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F);
            const int x1 = (x[i].qs[j] >>   4);
            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

void dequantize_row_q8_1(const block_q8_1* x, float* y, int64_t k)
{
    const int qk = QK8_1;
    const int64_t nb = k / qk;

    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < qk; ++j) y[i*qk + j] = x[i].qs[j]*d;
    }
}

void dequantize_row_q2_0(const block_q2_0* x, float* y, int64_t k)
{
    const int qk = QK2_0;
    const int64_t nb = k / qk;

    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < qk; ++j) {
            const int byte_index = j / 4;
            const int bit_offset = (j % 4) * 2;
            const uint8_t q = (x[i].qs[byte_index] >> bit_offset) & 0x03;
            // 00=-1, 01=0, 10=+1, 11=+2
            y[i*qk + j] = ((int)q - 1) * d;
        }
    }
}

void dequantize_row_pq2_0(const block_pq2_0* x, float* y, int64_t k)
{
    const int qk = 128;
    const int64_t nb = k / qk;

    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < qk; ++j) {
            const int byte_index = j / 4;
            const int bit_offset = (j % 4) * 2;
            const uint8_t q = (x[i].qs[byte_index] >> bit_offset) & 0x03;
            y[i*qk + j] = ((int)q - 1) * d;
        }
    }
}

void dequantize_row_mxfp4(const block_mxfp4* x, float* y, int64_t k)
{
    const int qk = QK_MXFP4;
    const int64_t nb = k / qk;

    for (int64_t i = 0; i < nb; i++) {
        const float d = ggml_e8m0_to_fp32(x[i].e);
        for (int j = 0; j < qk/2; ++j) {
            y[i*qk + j + 0   ] = kvalues_mxfp4_tbl[x[i].qs[j] & 0x0F] * d;
            y[i*qk + j + qk/2] = kvalues_mxfp4_tbl[x[i].qs[j] >>   4] * d;
        }
    }
}

/* TQ1_0 / PTQ1_0 share the base-3 trit codec; only the block width differs. */
static inline void tq1_decode_group(const uint8_t* qs, size_t count, const uint8_t* pow3,
                                    const float d, float*& y)
{
    for (size_t n = 0; n < 5; ++n) {
        for (size_t m = 0; m < count; ++m) {
            uint8_t q = qs[m] * pow3[n];
            int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
            *y++ = (float)(xi - 1) * d;
        }
    }
}

void dequantize_row_tq1_0(const block_tq1_0* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    const size_t nqs = sizeof(x->qs);

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        size_t j = 0;
        // 32-byte stages while they fit, then 16, then 8 — exactly as ggml does
        static const size_t stages[3] = {32, 16, 8};
        for (int s = 0; s < 3; ++s) {
            const size_t c = stages[s];
            for (; j + c <= nqs; j += c) tq1_decode_group(x[i].qs + j, c, pow3, d, y);
        }
        for (size_t n = 0; n < 4; ++n)
            for (size_t h = 0; h < sizeof(x->qh); ++h) {
                uint8_t q = x[i].qh[h] * pow3[n];
                int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                *y++ = (float)(xi - 1) * d;
            }
    }
}

void dequantize_row_ptq1_0(const block_ptq1_0* x, float* y, int64_t k)
{
    assert(k % 128 == 0);
    const int64_t nb = k / 128;
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    const size_t nqs = sizeof(x->qs);

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        size_t j = 0;
        static const size_t stages[3] = {32, 16, 8};
        for (int s = 0; s < 3; ++s) {
            const size_t c = stages[s];
            for (; j + c <= nqs; j += c) tq1_decode_group(x[i].qs + j, c, pow3, d, y);
        }
        for (size_t n = 0; n < 4; ++n)
            for (size_t h = 0; h < sizeof(x->qh); ++h) {
                uint8_t q = x[i].qh[h] * pow3[n];
                int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                *y++ = (float)(xi - 1) * d;
            }
    }
}

void dequantize_row_tq2_0(const block_tq2_0* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    const size_t nqs = sizeof(x->qs);

    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        for (size_t j = 0; j < nqs; j += 32)
            for (int l = 0; l < 4; ++l)
                for (int m = 0; m < 32; ++m) {
                    int8_t q = (x[i].qs[j + m] >> (l*2)) & 3;
                    *y++ = (float)(q - 1) * d;
                }
    }
}

void dequantize_row_q1_0(const block_q1_0* x, float* y, int64_t k)
{
    const int qk = QK1_0;
    const int64_t nb = k / qk;

    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const float neg_d = -d;
        for (int j = 0; j < qk; ++j) {
            const int byte_index = j / 8;
            const int bit_offset = j % 8;
            const uint8_t bit = (x[i].qs[byte_index] >> bit_offset) & 1;
            y[i*qk + j] = bit ? d : neg_d;
        }
    }
}

void dequantize_row_lq1_0(const block_lq1_0* x, float* y, int64_t k)
{
    const int nb = (int)(k / QK8_0);
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; j++) {
            int b = (x[i].qs[j >> 3] >> (j & 7)) & 1;
            y[i*QK8_0 + j] = d * (b? 1.0f : -1.0f);
        }
    }
}

void dequantize_row_lq1_G(const block_lq1_G* x, float* y, int64_t k)
{
    const int nb = (int)(k / 128);
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < 128; j++) {
            int b = (x[i].qs[j >> 3] >> (j & 7)) & 1;
            y[i*128 + j] = d * (b? 1.0f : -1.0f);
        }
    }
}

void dequantize_row_bf16(const uint16_t* x, float* y, int64_t k)
{
    for (int64_t i = 0; i < k; i++) {
        uint32_t bits = (uint32_t)x[i] << 16;
        memcpy(&y[i],&bits,4);
    }
}

void dequantize_row_i32(const int32_t* x, float* y, int64_t k)
{
    for (int64_t i = 0; i < k; i++) y[i] = (float)x[i];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Native dot products. Activation arrives as Q8_0 (32/block) or Q8_K
 * (256/block); weights are consumed in their packed form.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ---- Q4_0 : v = (nibble - 8) * d --------------------------------------- */
float dot_q8_0_q4_0(const block_q8_0* y, const block_q4_0* x, int n)
{
    const int nb = n / QK4_0;   // one Q8_0 block per 32 weights
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
        int sumi = 0;
        for (int j = 0; j < QK4_0/2; ++j) {
            sumi += (int)y[i].qs[j + 0]     * ((x[i].qs[j] & 0x0F) - 8);
            sumi += (int)y[i].qs[j + QK4_0/2] * ((x[i].qs[j] >> 4) - 8);
        }
        sumf += d * sumi;
    }
    return sumf;
}

/* ---- Q4_1 : v = nibble * d + m ------------------------------------------ */
float dot_q8_0_q4_1(const block_q8_0* y, const block_q4_1* x, int n)
{
    const int nb = n / QK4_1;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d  = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
        const float m  = fp16_to_fp32(x[i].m);
        int sumi = 0, qsum = 0;
        for (int j = 0; j < QK4_1; ++j) qsum += (int)y[i].qs[j];
        for (int j = 0; j < QK4_1/2; ++j) {
            sumi += (int)y[i].qs[j + 0]        * (x[i].qs[j] & 0x0F);
            sumi += (int)y[i].qs[j + QK4_1/2]  * (x[i].qs[j] >> 4);
        }
        sumf += d * sumi + m * qsum;
    }
    return sumf;
}

/* ---- Q8_1 : v = q * d, plus the precomputed sum shortcut ----------------- */
float dot_q8_0_q8_1(const block_q8_0* y, const block_q8_1* x, int n)
{
    const int nb = n / QK8_1;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
        int sumi = 0;
        for (int j = 0; j < QK8_1; ++j) sumi += (int)x[i].qs[j] * (int)y[i].qs[j];
        sumf += d * sumi;
    }
    return sumf;
}

/* ---- Q2_0 : code 0..3 -> (code-1) * d, 64 per block ---------------------- */
float dot_q8_0_q2_0(const block_q8_0* y, const block_q2_0* x, int n)
{
    const int nb = n / QK2_0;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const block_q8_0* y0 = y + i*2;      // 64 weights = 2 x 32-value Q8_0
        const block_q8_0* y1 = y0 + 1;
        int sumi = 0;
        for (int j = 0; j < QK2_0/4; ++j) {
            const uint8_t b = x[i].qs[j];
            for (int l = 0; l < 4; ++l) {
                const int q = (int)((b >> (l*2)) & 3) - 1;
                sumi += q * (int)y0->qs[j*4 + l];
                sumi += q * (int)y1->qs[j*4 + l];
            }
        }
        sumf += d * sumi;
    }
    return sumf;
}

/* ---- Q1_0 (upstream) : bit -> +d / -d, 128 per block -------------------- */
float dot_q8_0_q1_0(const block_q8_0* y, const block_q1_0* x, int n)
{
    const int nb = n / QK1_0;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const block_q8_0* yb = y + i*4;   // 128 = 4 * 32
        int sumi = 0;
        for (int k = 0; k < 4; k++) {
            const int bitbase = k * QK8_0;
            for (int j = 0; j < QK8_0; ++j) {
                const int bit = (x[i].qs[(bitbase + j) >> 3] >> ((bitbase + j) & 7)) & 1;
                sumi += (bit? 1 : -1) * (int)yb[k].qs[j];
            }
        }
        sumf += d * sumi;
    }
    return sumf;
}

/* ---- IQ4_NL : 16-entry codebook, split-half nibbles, 32 per block -------- */
void dequantize_row_iq4_nl(const block_iq4_nl* x, float* y, int64_t k)
{
    const int qk = QK4_NL;
    const int64_t nb = k / qk;
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < qk/2; ++j) {
            y[i*qk + j + 0   ] = d * kvalues_iq4nl_tbl[x[i].qs[j] & 0x0F];
            y[i*qk + j + qk/2] = d * kvalues_iq4nl_tbl[x[i].qs[j] >>   4];
        }
    }
}

/* One IQ4_NL block (32 values) lines up with exactly one Q8_0 activation block. */
float dot_q8_0_iq4_nl(const block_q8_0* y, const block_iq4_nl* x, int n)
{
    const int nb = n / QK4_NL;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
        int sumi = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            sumi += (int)kvalues_iq4nl_tbl[x[i].qs[j] & 0x0F] * (int)y[i].qs[j];
            sumi += (int)kvalues_iq4nl_tbl[x[i].qs[j] >>   4] * (int)y[i].qs[j + QK4_NL/2];
        }
        sumf += d * sumi;
    }
    return sumf;
}

/* ---- Q8_K activation paths (256 per super-block) ------------------------- */

/* Q2_0: 64 values/block, so 4 blocks per Q8_K super-block. */
float dot_q8_K_q2_0(const block_q8_K* y, const block_q2_0* x, int n)
{
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const int8_t* q8 = y[i].qs;
        for (int b = 0; b < 4; b++) {
            const block_q2_0* xb = x + i*4 + b;
            const float d = fp16_to_fp32(xb->d) * y[i].d;
            int sumi = 0;
            for (int j = 0; j < QK2_0/4; ++j) {
                const uint8_t byte = xb->qs[j];
                for (int l = 0; l < 4; ++l) {
                    const int q = (int)((byte >> (l*2)) & 3) - 1;
                    sumi += q * (int)q8[b*QK2_0 + j*4 + l];
                }
            }
            sumf += d * sumi;
        }
    }
    return sumf;
}

/* PQ2_0: identical codec, 128 values/block -> 2 blocks per super-block. */
float dot_q8_K_pq2_0(const block_q8_K* y, const block_pq2_0* x, int n)
{
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const int8_t* q8 = y[i].qs;
        for (int b = 0; b < 2; b++) {
            const block_pq2_0* xb = x + i*2 + b;
            const float d = fp16_to_fp32(xb->d) * y[i].d;
            int sumi = 0;
            for (int j = 0; j < 128/4; ++j) {
                const uint8_t byte = xb->qs[j];
                for (int l = 0; l < 4; ++l) {
                    const int q = (int)((byte >> (l*2)) & 3) - 1;
                    sumi += q * (int)q8[b*128 + j*4 + l];
                }
            }
            sumf += d * sumi;
        }
    }
    return sumf;
}

/* TQ2_0: 256 values/block, 2 bits each -> exactly one Q8_K super-block. */
float dot_q8_K_tq2_0(const block_q8_K* y, const block_tq2_0* x, int n)
{
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const int8_t* q8 = y[i].qs;
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        int sumi = 0;
        // qs layout: byte j holds elements {j*4+0..3} for each of the 4 l-planes
        for (int l = 0; l < 4; ++l)
            for (int j = 0; j < QK_K/4; ++j) {
                const int q = (int)((x[i].qs[j] >> (l*2)) & 3) - 1;
                sumi += q * (int)q8[l*(QK_K/4) + j];
            }
        sumf += d * sumi;
    }
    return sumf;
}

/* TQ1_0: decode trits on the fly, one weight block per Q8_K super-block. */
float dot_q8_K_tq1_0(const block_q8_K* y, const block_tq1_0* x, int n)
{
    const int nb = n / QK_K;
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    const size_t nqs = sizeof(x->qs), nqh = sizeof(x->qh);
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const block_tq1_0* xb = x + i;
        const float d = fp16_to_fp32(xb->d) * y[i].d;
        const int8_t* q8 = y[i].qs;
        int sumi = 0, idx = 0;

        size_t j = 0;
        static const size_t stages[3] = {32, 16, 8};
        for (int s = 0; s < 3; ++s) {
            const size_t c = stages[s];
            for (; j + c <= nqs; j += c)
                for (size_t nn = 0; nn < 5; ++nn)
                    for (size_t m = 0; m < c; ++m) {
                        const uint8_t q = xb->qs[j + m] * pow3[nn];
                        const int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                        sumi += (xi - 1) * (int)q8[idx++];
                    }
        }
        for (size_t nn = 0; nn < 4; ++nn)
            for (size_t h = 0; h < nqh; ++h) {
                const uint8_t q = xb->qh[h] * pow3[nn];
                const int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                sumi += (xi - 1) * (int)q8[idx++];
            }

        sumf += d * sumi;
    }
    return sumf;
}

float dot_q8_K_ptq1_0(const block_q8_K* y, const block_ptq1_0* x, int n)
{
    // 128 weight values per block -> two blocks per Q8_K super-block.
    const int nb = n / QK_K;
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    const size_t nqs = sizeof(x->qs), nqh = sizeof(x->qh);
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const int8_t* q8 = y[i].qs;
        for (int b = 0; b < 2; b++) {
            const block_ptq1_0* xb = x + i*2 + b;
            const float d = fp16_to_fp32(xb->d) * y[i].d;
            int sumi = 0, idx = b*128;

            size_t j = 0;
            static const size_t stages[3] = {32, 16, 8};
            for (int s = 0; s < 3; ++s) {
                const size_t c = stages[s];
                for (; j + c <= nqs; j += c)
                    for (size_t nn = 0; nn < 5; ++nn)
                        for (size_t m = 0; m < c; ++m) {
                            const uint8_t q = xb->qs[j + m] * pow3[nn];
                            const int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                            sumi += (xi - 1) * (int)q8[idx++];
                        }
            }
            for (size_t nn = 0; nn < 4; ++nn)
                for (size_t h = 0; h < nqh; ++h) {
                    const uint8_t q = xb->qh[h] * pow3[nn];
                    const int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                    sumi += (xi - 1) * (int)q8[idx++];
                }

            sumf += d * sumi;
        }
    }
    return sumf;
}

/* MXFP4: E8M0 scale + E2M1 nibbles, 32 per block -> 8 blocks per super-block */float dot_q8_K_mxfp4(const block_q8_K* y, const block_mxfp4* x, int n)
{
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const int8_t* q8 = y[i].qs;
        for (int b = 0; b < QK_K/QK_MXFP4; b++) {
            const block_mxfp4* xb = x + i*(QK_K/QK_MXFP4) + b;
            const float d = ggml_e8m0_to_fp32(xb->e) * y[i].d;
            int sumi = 0;
            for (int j = 0; j < QK_MXFP4/2; ++j) {
                sumi += kvalues_mxfp4_tbl[xb->qs[j] & 0x0F] * (int)q8[b*QK_MXFP4 + j];
                sumi += kvalues_mxfp4_tbl[xb->qs[j] >>   4] * (int)q8[b*QK_MXFP4 + j + QK_MXFP4/2];
            }
            sumf += d * sumi;
        }
    }
    return sumf;
}
