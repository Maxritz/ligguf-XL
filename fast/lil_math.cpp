/* LiGGUF - a small, dependency-free LLaMA inference engine with direct GGUF support
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#include <assert.h>
#include <string.h>
#include "common.h"
#include "lil_math.h"
#include "breadcrumb.h"
#include "lil_quants.h"

using namespace std;

float fp1632_lut[65536];

static qtensor m_xq8;
static ktensor m_xqK;

// Acceleration stuff
#if defined(__AVX2__)
#include "arch_avx2.h"
#endif
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include "arch_neon.h"
#endif

static inline float fp32_from_bits(uint32_t w)
{
    float r;
    memcpy(&r,&w,4);
    return r;
}

static inline uint32_t fp32_to_bits(float f)
{
    uint32_t r;
    memcpy(&r,&f,4);
    return r;
}

static float fp16_to_fp32_impl(uint16_t h)
{
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));

    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline uint16_t fp32_to_fp16(float f)
{
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));

    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) bias = UINT32_C(0x71000000);

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}

void fp1632_init()
{
    for (int i = 0; i < 65536; i++) fp1632_lut[i] = fp16_to_fp32_impl((uint16_t)i);
}

void f32_to_f16_row(const float* x, gguf_half* y, int n)
{
    for (int i = 0; i < n; i++) y[i] = fp32_to_fp16(x[i]);
}

static inline void f16_to_f32_row(const gguf_half* x, float* y, int n)
{
    for (int i = 0; i < n; i++) y[i] = fp16_to_fp32(x[i]);
}

void dequantize_row_q8_0(const block_q8_0* x, float* y, int64_t k)
{
    const int nb = (int)(k / QK8_0);

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; j++) y[i*QK8_0 + j] = x[i].qs[j] * d;
    }
}

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4);
    }
}

void dequantize_row_q2_K(const block_q2_K* x, float* y, int64_t k)
{
    const int nb = (int)(k / QK_K);

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);
        const uint8_t* q = x[i].qs;

        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = x[i].scales[is++];
                float dl = d * (sc & 0xF), ml = min * (sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF), ml = min * (sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

void dequantize_row_q3_K(const block_q3_K* x, float* y, int64_t k)
{
    // Q3_K stores 256 values as:
    // - 2-bit payload in qs[64]
    // - one extra sign/high-bit plane in hmask[32]
    // - 16 signed scales packed into scales[12]
    // - one fp16 block scale d
    const int nb = (int)(k / QK_K);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {
        const float d_all = fp16_to_fp32(x[i].d);
        const uint8_t* q = x[i].qs;
        const uint8_t* hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

void dequantize_row_q4_K(const block_q4_K* x, float* y, int64_t k)
{
    // Q4_K stores 256 values as 4-bit payloads in qs[128], plus 8 scale/min pairs
    // packed into scales[12]. d scales the values, dmin scales the per-subblock offsets.
    const int nb = (int)(k / QK_K);

    for (int i = 0; i < nb; i++) {
        const uint8_t* q = x[i].qs;
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

void dequantize_row_q5_K(const block_q5_K* x, float* y, int64_t k)
{
    // Q5_K is Q4_K plus one extra bit plane in qh[32], so each value is 5 bits
    // with the same packed scale/min layout and shared block scales d/dmin.
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/* Q5_0/Q5_1 dequantizers */
/* qh[4] is bit-packed: bit j of element j lives in qh[j>>3] >> (j&7) */
void dequantize_row_q5_0(const block_q5_0* x, float* y, int64_t k)
{
    const int nb = (int)(k / QK8_0);
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));
        /* qs[j] low nibble -> element j, high nibble -> element j + qk/2.
         * Bit j of qh is the 5th bit of element j. Value is 5-bit signed. */
        for (int j = 0; j < QK8_0/2; ++j) {
            const int xh0 = (int)(qh >> j)         & 1;
            const int xh1 = (int)(qh >> (j + 16))  & 1;
            const int x0 = ((x[i].qs[j] & 0x0F) | (xh0 << 4)) - 16;
            const int x1 = ((x[i].qs[j] >>   4) | (xh1 << 4)) - 16;
            y[i*QK8_0 + j + 0     ] = x0*d;
            y[i*QK8_0 + j + QK8_0/2] = x1*d;
        }
    }
}

void dequantize_row_q5_1(const block_q5_1* x, float* y, int64_t k)
{
    const int nb = (int)(k / QK8_0);
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const float m = fp16_to_fp32(x[i].m);
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));
        for (int j = 0; j < QK8_0/2; ++j) {
            const int xh0 = (int)(qh >> j)         & 1;
            const int xh1 = (int)(qh >> (j + 16))  & 1;
            const int x0 = (x[i].qs[j] & 0x0F) | (xh0 << 4);
            const int x1 = (x[i].qs[j] >>   4) | (xh1 << 4);
            y[i*QK8_0 + j + 0     ] = x0*d + m;
            y[i*QK8_0 + j + QK8_0/2] = x1*d + m;
        }
    }
}

void dequantize_row_q6_K(const block_q6_K* x, float* y, int64_t k)
{
    // Q6_K stores low 4 bits in ql[128], high 2 bits in qh[64], one signed scale
    // per 16 values in scales[16], and a shared fp16 block scale d.
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dequantize_row(gguf_type t, const void* in, float* out, int64_t n)
{
    switch (t) {
    case F32:  memcpy(out,in,n*sizeof(float)); break;
    case F16:  f16_to_f32_row((const gguf_half*)in,out,(int)n); break;
    case BF16: dequantize_row_bf16((const uint16_t*)in,out,n); break;
    case F64:  for (int64_t i=0;i<n;i++) out[i]=(float)((const double*)in)[i]; break;
    case I8:   for (int64_t i=0;i<n;i++) out[i]=(float)((const int8_t*)in)[i]; break;
    case I16:  for (int64_t i=0;i<n;i++) out[i]=(float)((const int16_t*)in)[i]; break;
    case I32:  dequantize_row_i32((const int32_t*)in,out,n); break;
    case I64:  for (int64_t i=0;i<n;i++) out[i]=(float)((const int64_t*)in)[i]; break;
    case Q8_0: dequantize_row_q8_0((block_q8_0*)in,out,n); break;
    case Q8_1: dequantize_row_q8_1((const block_q8_1*)in,out,n); break;
    case Q4_0: dequantize_row_q4_0((const block_q4_0*)in,out,n); break;
    case Q4_1: dequantize_row_q4_1((const block_q4_1*)in,out,n); break;
    case Q2_0: dequantize_row_q2_0((const block_q2_0*)in,out,n); break;
    case IQ4_NL:dequantize_row_iq4_nl((const block_iq4_nl*)in,out,n); break;
    case PQ2_0:dequantize_row_pq2_0((const block_pq2_0*)in,out,n); break;
    case MXFP4:dequantize_row_mxfp4((const block_mxfp4*)in,out,n); break;
    case TQ1_0:dequantize_row_tq1_0((const block_tq1_0*)in,out,n); break;
    case PTQ1_0:dequantize_row_ptq1_0((const block_ptq1_0*)in,out,n); break;
    case TQ2_0:dequantize_row_tq2_0((const block_tq2_0*)in,out,n); break;
    case Q1_0: dequantize_row_q1_0((const block_q1_0*)in,out,n); break;
    case LQ1_0:dequantize_row_lq1_0((const block_lq1_0*)in,out,n); break;
    case LQ1_G:dequantize_row_lq1_G((const block_lq1_G*)in,out,n); break;
    case Q2_K: dequantize_row_q2_K((block_q2_K*)in,out,n); break;
    case Q3_K: dequantize_row_q3_K((block_q3_K*)in,out,n); break;
    case Q4_K: dequantize_row_q4_K((block_q4_K*)in,out,n); break;
    case Q5_K: dequantize_row_q5_K((block_q5_K*)in,out,n); break;
    case Q6_K: dequantize_row_q6_K((block_q6_K*)in,out,n); break;
    case Q5_0: dequantize_row_q5_0((const block_q5_0*)in,out,n); break;
    case Q5_1: dequantize_row_q5_1((const block_q5_1*)in,out,n); break;
    default:
        ERR("Unsupported tensor type %d",t);
        abort();
    }
}

float vec_dot_f32(const float* a, const float* b, int size)
{
    return dot_f32(a,b,size);
}

float vec_dot_f16_f32(const gguf_half* a, const float* b, int size)
{
    return dot_f16_f32(a,b,size);
}

void quantize_q8_0(qtensor &y, const ftensor &x)
{
    const int nb = (int)(x.size() / QK8_0);
    if ((int)y.size() < nb) y.resize(nb);

    MULTITHREAD
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = max(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = fp32_to_fp16(d);

        for (int j = 0; j < QK8_0; j++)
            y[i].qs[j] = (int8_t)roundf(x[i * QK8_0 + j] * id);
    }
}

static inline int nearest_int(float f)
{
    return f < 0 ? (int)(f - 0.5f) : (int)(f + 0.5f);
}

void quantize_q8_K(ktensor &y, const ftensor &x)
{
    const int n = (int)x.size();
    const int nb = (n + QK_K - 1) / QK_K;   /* round up to cover remainders */

    if ((int)y.size() < nb) y.resize(nb);

    MULTITHREAD
    for (int i = 0; i < nb; i++) {
        const float* p = x.data() + i * QK_K;
        int ncur = (i == nb - 1) ? (n - i * QK_K) : QK_K;
        float max = 0.0f;
        float amax = 0.0f;
        for (int j = 0; j < ncur; j++) {
            float ax = fabsf(p[j]);
            if (ax > amax) {
                amax = ax;
                max = p[j];
            }
        }
        if (!amax) {
            y[i].d = 0.0f;
            memset(y[i].qs,0,QK_K);
            memset(y[i].bsums,0,sizeof(y[i].bsums));
            continue;
        }

        const float iscale = -127.0f / max;
        memset(y[i].qs, 0, QK_K);  /* zero-pad remainder */
        for (int j = 0; j < ncur; j++) {
            int v = nearest_int(iscale * p[j]);
            y[i].qs[j] = (int8_t)min(127, max(-128, v));
        }
        for (int j = 0; j < QK_K/16; j++) {
            int sum = 0;
            int jmax = (j+1)*16 <= ncur ? 16 : (ncur - j*16);
            for (int ii = 0; ii < jmax; ii++) sum += (int)y[i].qs[j*16 + ii];
            y[i].bsums[j] = (int16_t)sum;
        }
        y[i].d = 1.0f / iscale;
    }
}

#ifndef ARCH_ACCEL
float inline dot_f32(const float* a, const float* b, int size)
{
    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += a[i] * b[i];
    return sum;
}

float inline dot_f16_f32(const gguf_half* a, const float* b, int size)
{
    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += fp16_to_fp32(a[i]) * b[i];
    return sum;
}

float inline dot_q8_0_q8_0(const block_q8_0* x, const block_q8_0* y, int n)
{
    const int nb = n / QK8_0;
    float acc = 0.0f;
    for (int s,j,i = 0; i < nb; i++) {
        s = 0;
        for (j = 0; j < QK8_0; j++) s += x[i].qs[j] * y[i].qs[j];
        acc += fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d) * s;
    }
    return acc;
}

float inline dot_q8_0_lq1_0(const block_q8_0* y, const block_lq1_0* x, int n)
{
    const int nb = n / QK8_0;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; j++) {
            int xi = ((x[i].qs[j >> 3] >> (j & 7)) & 1)? 1 : -1;
            sumi += xi * y[i].qs[j];
        }
        sumf += fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d) * sumi;
    }

    return sumf;
}

float inline dot_q8_0_lq1_G(const block_q8_0* y, const block_lq1_G* x, int n)
{
    const int nb = n / 128;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d0 = fp16_to_fp32(x[i].d);
        float sumi = 0.0f;

        for (int k = 0; k < 4; k++) {
            const block_q8_0* yb = y + i * 4 + k;
            int sumib = 0;
            for (int j = 0; j < QK8_0; j++) {
                int bit = k * QK8_0 + j;
                int xi = ((x[i].qs[bit >> 3] >> (bit & 7)) & 1)? 1 : -1;
                sumib += xi * yb->qs[j];
            }
            sumi += fp16_to_fp32(yb->d) * sumib;
        }

        sumf += d0 * sumi;
    }

    return sumf;
}

float inline dot_q8_K_q3_K(const block_q8_K* y, const block_q3_K* x, int n)
{
    // Rebuild the signed Q3 values into aux8[], expand the packed per-16 scales,
    // then accumulate the dot in 16-value chunks against the runtime Q8_K activation.
    const int nb = n / QK_K;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums,0,sizeof(sums));
    uint32_t auxs[4];
    const int8_t* scales = (const int8_t*)auxs;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t* q3 = x[i].qs;
        const uint8_t* hm = x[i].hmask;
        const int8_t* q8 = y[i].qs;
        memset(aux32,0,sizeof(aux32));
        int8_t* a = aux8;
        uint8_t m = 1;
        // Expand the 2-bit payload plus high-bit mask into signed 3-bit values.
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; l++) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; l++) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32;
            m <<= 1;
            for (int l = 0; l < 32; l++) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; l++) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32;
            m <<= 1;
            for (int l = 0; l < 32; l++) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; l++) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32;
            m <<= 1;
            for (int l = 0; l < 32; l++) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; l++) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32;
            m <<= 1;
            q3 += 32;
        }

        a = aux8;
        memcpy(auxs,x[i].scales,12);

        // Expand the packed sub-block scales to one signed scale per 16 values.
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        for (int j = 0; j < QK_K/16; j++) {
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8;
            a += 8;
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8;
            a += 8;
        }

        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; l++) sums[l] += d * aux32[l];
    }

    for (int l = 0; l < 8; l++) sumf += sums[l];
    return sumf;
}

float inline dot_q8_K_q2_K(const block_q8_K* y, const block_q2_K* x, int n)
{
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t* q2 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        const uint8_t* sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < QK_K/16; j++) summs += y[i].bsums[j] * (sc[j] >> 4);

        const float dall = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = y[i].d * fp16_to_fp32(x[i].dmin);

        int isum = 0;
        int is = 0;
        for (int k = 0; k < QK_K/128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l = 0; l < 16; l++) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;

                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; l++) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;

                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }

        sumf += dall * isum - dmin * summs;
    }

    return sumf;
}

float inline dot_q8_K_q4_K(const block_q8_K* y, const block_q4_K* x, int n)
{
    // Walk the two 32-value halves inside each 64-value chunk, dot the activation
    // against each nibble stream, then apply the chunk scale and min correction.
    const int nb = n / QK_K;
    float acc = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t* q = x[i].qs;
        const int8_t* a = y[i].qs;
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        const float dmin = fp16_to_fp32(x[i].dmin) * y[i].d;
        int is = 0;

        // Each 64-value chunk is two 32-value dots with separate scale/min pairs.
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0,x[i].scales,&sc,&m);
            int dot1 = 0;
            for (int l = 0; l < 32; l++) dot1 += a[l] * (q[l] & 0xF);
            acc += d * sc * dot1 - dmin * m * (y[i].bsums[is*2 + 0] + y[i].bsums[is*2 + 1]);

            get_scale_min_k4(is + 1,x[i].scales,&sc,&m);
            int dot2 = 0;
            for (int l = 0; l < 32; l++) dot2 += a[l + 32] * (q[l] >> 4);
            acc += d * sc * dot2 - dmin * m * (y[i].bsums[is*2 + 2] + y[i].bsums[is*2 + 3]);

            q += 32;
            a += 64;
            is += 2;
        }
    }

    return acc;
}

float inline dot_q8_K_q5_K(const block_q8_K* y, const block_q5_K* x, int n)
{
    // Rebuild the 5-bit payload into aux8[], expand the packed scale/min tables,
    // accumulate scaled dot products, then subtract the min contribution via bsums.
    const int nb = n / QK_K;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t* scales = (const uint8_t*)&utmp[0];
    const uint8_t* mins = (const uint8_t*)&utmp[2];
    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums,0,sizeof(sums));
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t* q4 = x[i].qs;
        const uint8_t* hm = x[i].qh;
        const int8_t* q8 = y[i].qs;
        memset(aux32,0,sizeof(aux32));
        int8_t* a = aux8;
        uint8_t m = 1;

        // Rebuild the 5-bit values before applying scale and min corrections.
        for (int j = 0; j < QK_K/64; j++) {
            for (int l = 0; l < 32; l++) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; l++) a[l] += (hm[l] & m ? 16 : 0);
            a += 32;
            m <<= 1;
            for (int l = 0; l < 32; l++) a[l] = (int8_t)(q4[l] >> 4);
            for (int l = 0; l < 32; l++) a[l] += (hm[l] & m ? 16 : 0);
            a += 32;
            m <<= 1;
            q4 += 32;
        }

        memcpy(utmp,x[i].scales,12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; j++) sumi += y[i].bsums[j] * mins[j/2];

        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; j++) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }

        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; l++) sums[l] += d * aux32[l];
        sumf -= fp16_to_fp32(x[i].dmin) * y[i].d * sumi;
    }

    for (int l = 0; l < 8; l++) sumf += sums[l];
    return sumf;
}

float inline dot_q8_K_q6_K(const block_q8_K* y, const block_q6_K* x, int n)
{
    const int nb = n / QK_K;
    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8];
    int32_t aux32[8];
    memset(sums,0,sizeof(sums));
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t* q4 = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* q8 = y[i].qs;
        memset(aux32,0,sizeof(aux32));
        int8_t* a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; l++) {
                a[l + 0] = (int8_t)((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a += 128;
            q4 += 64;
            qh += 32;
        }

        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; j++) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; l++) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; l++) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }

        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; l++) sums[l] += d * aux32[l];
    }

    for (int l = 0; l < 8; l++) sumf += sums[l];
    return sumf;
}
#endif /*ARCH_ACCEL*/

/* ═══════════════════════════════════════════════════════════════════════════
 * Single row-dot dispatcher shared by matmul / matmul2 / matmul3.
 * Keeps every quantization type in exactly one place, so adding a codec cannot
 * drift between the three call sites.
 *
 * `n` is the row width (number of weights). Which activation buffer is used is
 * decided by the weight's block width:
 *   QK8_0-wide types  -> qtensor (Q8_0)
 *   QK_K-wide types   -> ktensor (Q8_K)
 * ═══════════════════════════════════════════════════════════════════════════ */
static inline bool weight_needs_q8_0(gguf_type t)
{
    switch (t) {
        case Q8_0: case Q8_1: case Q4_0: case Q4_1:
        case Q5_0: case Q5_1: case Q2_0: case Q1_0:
        case IQ4_NL:
        case LQ1_0: case LQ1_G:
            return true;
        default:
            return false;
    }
}

static inline bool weight_needs_q8_K(gguf_type t)
{
    switch (t) {
        case Q2_K: case Q3_K: case Q4_K: case Q5_K: case Q6_K:
        case PQ2_0: case TQ1_0: case PTQ1_0: case TQ2_0: case MXFP4:
            return true;
        default:
            return false;
    }
}

/* Both activation forms, resolved once per matmul call. */
struct actbuf {
    const block_q8_0* q8;
    const block_q8_K* qk;
};

static inline float row_dot(gguf_type t, const uint8_t* wp, const ftensor* src,
                            const block_q8_0* q8, const block_q8_K* qk, int n)
{
    /* Diagnostic: force the slow, always-correct dequantize + float-FMA path.
     * Set FLOAT_DOT=1. Used to separate "quant codec" bugs from graph bugs. */
    static int fdmode = -1;
    if (fdmode < 0) { const char* e = getenv("FLOAT_DOT"); fdmode = e ? atoi(e) : 0; }
    if (fdmode) {
        static thread_local std::vector<float> tmp;
        if ((int)tmp.size() < n) tmp.resize(n);
        dequantize_row(t, wp, tmp.data(), n);
        return dot_f32(tmp.data(), src->data(), n);
    }

    switch (t) {
        case F32:  return dot_f32((const float*)wp,src->data(),n);
        case F16:  return dot_f16_f32((const gguf_half*)wp,src->data(),n);
        case Q8_0: assert(q8); return dot_q8_0_q8_0(q8,(const block_q8_0*)wp,n);
        case Q8_1: assert(q8); return dot_q8_0_q8_1(q8,(const block_q8_1*)wp,n);
        case Q4_0: assert(q8); return dot_q8_0_q4_0(q8,(const block_q4_0*)wp,n);
        case Q4_1: assert(q8); return dot_q8_0_q4_1(q8,(const block_q4_1*)wp,n);
        case Q5_0: assert(q8); return dot_q8_0_q5_0(q8,(const block_q5_0*)wp,n);
        case Q5_1: assert(q8); return dot_q8_0_q5_1(q8,(const block_q5_1*)wp,n);
        case Q2_0: assert(q8); return dot_q8_0_q2_0(q8,(const block_q2_0*)wp,n);
        case IQ4_NL:assert(q8); return dot_q8_0_iq4_nl(q8,(const block_iq4_nl*)wp,n);
        case Q1_0: assert(q8); return dot_q8_0_q1_0(q8,(const block_q1_0*)wp,n);
        case LQ1_0:assert(q8); return dot_q8_0_lq1_0(q8,(const block_lq1_0*)wp,n);
        case LQ1_G:assert(q8); return dot_q8_0_lq1_G(q8,(const block_lq1_G*)wp,n);
        case Q2_K: assert(qk); return dot_q8_K_q2_K(qk,(const block_q2_K*)wp,n);
        case Q3_K: assert(qk); return dot_q8_K_q3_K(qk,(const block_q3_K*)wp,n);
        case Q4_K: assert(qk); return dot_q8_K_q4_K(qk,(const block_q4_K*)wp,n);
        case Q5_K: assert(qk); return dot_q8_K_q5_K(qk,(const block_q5_K*)wp,n);
        case Q6_K: assert(qk); return dot_q8_K_q6_K(qk,(const block_q6_K*)wp,n);
        case PQ2_0:assert(qk); return dot_q8_K_pq2_0(qk,(const block_pq2_0*)wp,n);
        case TQ1_0:assert(qk); return dot_q8_K_tq1_0(qk,(const block_tq1_0*)wp,n);
        case PTQ1_0:assert(qk); return dot_q8_K_ptq1_0(qk,(const block_ptq1_0*)wp,n);
        case TQ2_0:assert(qk); return dot_q8_K_tq2_0(qk,(const block_tq2_0*)wp,n);
        case MXFP4:assert(qk); return dot_q8_K_mxfp4(qk,(const block_mxfp4*)wp,n);
        default:
            ERR("Unsupported matmul type %d",t);
            abort();
    }
}

/* Resolve the activations the given weight types need, quantizing on demand from
 * `src` only when the caller did not already supply the right form. Done once per
 * matmul call, never inside the row loop. */
static inline actbuf prepare_activation(gguf_type t0, gguf_type t1, gguf_type t2,
                                        const ftensor* src,
                                        const ktensor* qxk, const qtensor* qx8)
{
    actbuf a; a.q8 = NULL; a.qk = NULL;

    const bool need8 = weight_needs_q8_0(t0) || weight_needs_q8_0(t1) || weight_needs_q8_0(t2);
    const bool needK = weight_needs_q8_K(t0) || weight_needs_q8_K(t1) || weight_needs_q8_K(t2);

    if (need8) {
        if (qx8) a.q8 = qx8->data();
        else { assert(src); quantize_q8_0(m_xq8,*src); a.q8 = m_xq8.data(); }
    }
    if (needK) {
        if (qxk) a.qk = qxk->data();
        else { assert(src); quantize_q8_K(m_xqK,*src); a.qk = m_xqK.data(); }
    }
    return a;
}

// out: destination vector with d rows
// src: float activation to quantize on demand when the weight is Q8_0/Q8_K-family
// qxk/qx8: pre-quantized activation in Q8_K or Q8_0 form (optional)
// w: source weight tensor row layout
// n: input width, d: output width
void matmul(ftensor &out, const ftensor* src, const ktensor* qxk, const qtensor* qx8, wtensor w, int n, int d)
{
    STRACE_MATMUL("matmul", n, d);
    if (!w.ptr) { ERR("matmul: NULL weight tensor (n=%d d=%d)",n,d); abort(); }
    if (n <= 0 || d <= 0) { ERR("matmul: bad dims n=%d d=%d",n,d); abort(); }

    const actbuf a = prepare_activation(w.type,w.type,w.type,src,qxk,qx8);

    MULTITHREAD
    for (int r = 0; r < d; r++)
        out[r] = row_dot(w.type,w.ptr + (uint64_t)r * w.rsz,src,a.q8,a.qk,n);
}

// Same contract as matmul(), but computes two same-shape outputs in one row loop.
// The two weight matrices need not share a type; both activation forms are resolved.
void matmul2(ftensor &out0, ftensor &out1, const ftensor* src, const ktensor* qxk, const qtensor* qx8, wtensor w0, wtensor w1, int n, int d)
{
    if (!w0.ptr || !w1.ptr) { ERR("matmul2: NULL weight tensor"); abort(); }
    if (n <= 0 || d <= 0) { ERR("matmul2: bad dims n=%d d=%d",n,d); abort(); }

    const actbuf a = prepare_activation(w0.type,w1.type,w1.type,src,qxk,qx8);

    MULTITHREAD
    for (int r = 0; r < d; r++) {
        out0[r] = row_dot(w0.type,w0.ptr + (uint64_t)r * w0.rsz,src,a.q8,a.qk,n);
        out1[r] = row_dot(w1.type,w1.ptr + (uint64_t)r * w1.rsz,src,a.q8,a.qk,n);
    }
}

// Q, K and V in one row loop. d0/d1/d2 are the three output widths, which differ
// whenever the Q projection also carries a gate (Qwen3.5 hybrid models).
void matmul3(ftensor &out0, ftensor &out1, ftensor &out2, const ftensor* src, const ktensor* qxk, const qtensor* qx8, wtensor w0, wtensor w1, wtensor w2, int n, int d0, int d1, int d2)
{
    if (!w0.ptr || !w1.ptr || !w2.ptr) {
        ERR("matmul3: NULL weight tensor (q=%p k=%p v=%p)",(void*)w0.ptr,(void*)w1.ptr,(void*)w2.ptr);
        abort();
    }
    if (n <= 0 || d0 <= 0 || d1 <= 0 || d2 <= 0) {
        ERR("matmul3: bad dims n=%d d0=%d d1=%d d2=%d",n,d0,d1,d2);
        abort();
    }

    const actbuf a = prepare_activation(w0.type,w1.type,w2.type,src,qxk,qx8);

    int d = d0;
    if (d1 > d) d = d1;
    if (d2 > d) d = d2;

    MULTITHREAD
    for (int r = 0; r < d; r++) {
        if (r < d0) out0[r] = row_dot(w0.type,w0.ptr + (uint64_t)r * w0.rsz,src,a.q8,a.qk,n);
        if (r < d1) out1[r] = row_dot(w1.type,w1.ptr + (uint64_t)r * w1.rsz,src,a.q8,a.qk,n);
        if (r < d2) out2[r] = row_dot(w2.type,w2.ptr + (uint64_t)r * w2.rsz,src,a.q8,a.qk,n);
    }
}

/* ── numerical self-test ──────────────────────────────────────────────────── */

static void dequant_q8_0_blocks(const block_q8_0* x, int nb, float* y)
{
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; j++) y[i*QK8_0 + j] = x[i].qs[j] * d;
    }
}

static void dequant_q8_K_blocks(const block_q8_K* x, int nb, float* y)
{
    for (int i = 0; i < nb; i++) {
        const float d = x[i].d;
        for (int j = 0; j < QK_K; j++) y[i*QK_K + j] = x[i].qs[j] * d;
    }
}

/* Fused dot vs. a fully dequantized double-precision reference. */
double selftest_row(gguf_type t, const uint8_t* wptr, int n, int trials){
    std::vector<float> wr(n);
    dequantize_row(t, wptr, wr.data(), n);

    double worst = 0.0;
    uint32_t seed = 1234567u;
    for (int tr = 0; tr < trials; tr++) {
        ftensor act(n);
        for (int i = 0; i < n; i++) {
            seed = seed * 1664525u + 1013904223u;
            act[i] = ((float)(seed >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
        }

        const actbuf a = prepare_activation(t,t,t,&act,NULL,NULL);
        const float got = row_dot(t, wptr, &act, a.q8, a.qk, n);

        std::vector<float> ar(n);
        if (weight_needs_q8_0(t))      dequant_q8_0_blocks(a.q8, n/QK8_0, ar.data());
        else if (weight_needs_q8_K(t)) dequant_q8_K_blocks(a.qk, n/QK_K, ar.data());
        else for (int i = 0; i < n; i++) ar[i] = act[i];

        double ref = 0.0;
        for (int i = 0; i < n; i++) ref += (double)wr[i] * (double)ar[i];

        const double denom = fabs(ref) > 1e-12 ? fabs(ref) : 1.0;
        const double err = fabs(ref - (double)got) / denom;
        if (err > worst) worst = err;
    }
    return worst;
}

/* Round-trip float -> f16 -> float; returns worst relative error. */
double selftest_fp16()
{
    double worst = 0.0;
    for (int i = -2000; i <= 2000; i++) {
        const float v = (float)i * 0.001f + 0.000123f;
        const float r = fp16_to_fp32(fp32_to_fp16(v));
        const double denom = fabsf(v) > 1e-4f ? fabsf(v) : 1e-4;
        const double e = fabs((double)r - (double)v) / denom;
        if (e > worst) worst = e;
    }
    return worst;
}

void rope(ftensor& x, int n_heads, int pos, const gguf_model &mdl, const ftensor &rope_freq)
{
    assert((int)x.size() == n_heads * mdl.head_dim);
    int rope_dim = mdl.rope_dim;
    if (rope_dim > mdl.head_dim) rope_dim = mdl.head_dim;
    if (rope_dim & 1) rope_dim--;

    /* LLaMA-style "half-split" RoPE (ggml GGML_ROPE_TYPE_NORM, is_neox = false):
     * the two rotated components are (i) and (i + rope_dim/2), NOT adjacent. */
    const int half = rope_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float* v = x.data() + h * mdl.head_dim;
        for (int m = 0; m < half; m++) {
            const float ang = pos * rope_freq[m];
            const float c = cosf(ang), s = sinf(ang);
            const float x0 = v[m];
            const float x1 = v[m + half];
            v[m]        = x0 * c - x1 * s;
            v[m + half] = x0 * s + x1 * c;
        }
    }
}

void rmsnorm(ftensor &out, const ftensor &x, float* w, int size, float epsilon)
{
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = ss / (float)size + epsilon;
    ss = 1.0f / sqrtf(ss);
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * w[i];
}

void rmsnorm_inplace(float* x, float* w, int size, float epsilon)
{
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)size + epsilon);
    for (int i = 0; i < size; i++) x[i] = x[i] * ss * w[i];
}

void l2norm(ftensor &x)
{
    float ss = 0.0f;
    for (int i = 0; i < (int)x.size(); i++) ss += x[i] * x[i];
    if (ss <= 0.0f) return;
    ss = 1.0f / sqrtf(ss);
    for (int i = 0; i < (int)x.size(); i++) x[i] *= ss;
}

void layernorm(ftensor &out, const ftensor &x, const float* w, const float* b, int size, float epsilon)
{
    double mean = 0.0;
    double var = 0.0;
    for (int i = 0; i < size; i++) mean += x[i];
    mean /= size;
    for (int i = 0; i < size; i++) {
        const double d = x[i] - mean;
        var += d * d;
    }
    var /= size;

    const float scale = (float)(1.0 / sqrt(var + epsilon));
    for (int i = 0; i < size; i++) out[i] = (float)((x[i] - mean) * scale * w[i] + b[i]);
}

void softmax(float* x, int size)
{
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < size; i++) x[i] /= sum;
}

void gelu(ftensor &x)
{
    for (int i = 0; i < (int)x.size(); i++) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}
