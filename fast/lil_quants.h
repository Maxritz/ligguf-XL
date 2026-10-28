/* LiGGUF - additional quantization codecs (dequant + native dot products)
 * Reference implementations follow ggml/src/ggml-quants.c and ggml-common.h
 * from ggml-org/llama.cpp, plus the PrismML fork for PQ2_0 / PTQ1_0.
 *
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#pragma once

#include "lil_gguf.h"

/* ── dequantization (used for embeddings / norms / any row we need as float) ── */

void dequantize_row_q4_0(const block_q4_0* x, float* y, int64_t k);
void dequantize_row_q4_1(const block_q4_1* x, float* y, int64_t k);
void dequantize_row_q8_1(const block_q8_1* x, float* y, int64_t k);
void dequantize_row_q2_0(const block_q2_0* x, float* y, int64_t k);
void dequantize_row_pq2_0(const block_pq2_0* x, float* y, int64_t k);
void dequantize_row_mxfp4(const block_mxfp4* x, float* y, int64_t k);
void dequantize_row_tq1_0(const block_tq1_0* x, float* y, int64_t k);
void dequantize_row_ptq1_0(const block_ptq1_0* x, float* y, int64_t k);
void dequantize_row_tq2_0(const block_tq2_0* x, float* y, int64_t k);
void dequantize_row_q1_0(const block_q1_0* x, float* y, int64_t k);
void dequantize_row_lq1_0(const block_lq1_0* x, float* y, int64_t k);
void dequantize_row_lq1_G(const block_lq1_G* x, float* y, int64_t k);
void dequantize_row_bf16(const uint16_t* x, float* y, int64_t k);
void dequantize_row_i32(const int32_t* x, float* y, int64_t k);
void dequantize_row_iq4_nl(const block_iq4_nl* x, float* y, int64_t k);

/* ── native dot products: activation x weight, no dequantization ─────────── */

float dot_q8_0_iq4_nl(const block_q8_0* y, const block_iq4_nl* x, int n);
float dot_q8_0_q4_0(const block_q8_0* y, const block_q4_0* x, int n);
float dot_q8_0_q4_1(const block_q8_0* y, const block_q4_1* x, int n);
float dot_q8_0_q8_1(const block_q8_0* y, const block_q8_1* x, int n);
float dot_q8_0_q2_0(const block_q8_0* y, const block_q2_0* x, int n);
float dot_q8_0_q1_0(const block_q8_0* y, const block_q1_0* x, int n);

float dot_q8_K_q2_0(const block_q8_K* y, const block_q2_0* x, int n);
float dot_q8_K_pq2_0(const block_q8_K* y, const block_pq2_0* x, int n);
float dot_q8_K_tq1_0(const block_q8_K* y, const block_tq1_0* x, int n);
float dot_q8_K_ptq1_0(const block_q8_K* y, const block_ptq1_0* x, int n);
float dot_q8_K_tq2_0(const block_q8_K* y, const block_tq2_0* x, int n);
float dot_q8_K_mxfp4(const block_q8_K* y, const block_mxfp4* x, int n);

/* ── helpers shared with the Hadamard fold ────────────────────────────────── */

float ggml_e8m0_to_fp32(uint8_t e);
extern const int8_t kvalues_mxfp4_tbl[16];
extern const int8_t kvalues_iq4nl_tbl[16];
