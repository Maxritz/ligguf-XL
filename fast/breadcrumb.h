/* LiGGUF - lightweight breadcrumb (Sherlock) tracing
 *
 * Two independent env-gated layers:
 *
 *   STRACE=1        — debug breadcrumbs (granular, per-layer/per-token)
 *                        BREADCRUMB comp=attention evt=dispatch layer=3
 *
 *   PERF_TRACE=1    — lightweight timing (phase-level only, RAII)
 *                        PERF_SCOPE comp=inference evt=token tok=42 ms=12.3
 *
 * Design rules:
 *   • STRACE can trace anything — it is debug instrumentation; perf cost
 *     is acceptable.
 *   • PERF_TRACE must be _light_ — only at phase boundaries (never inside
 *     per-layer or per-head loops) to avoid perturbing measurements.
 *   • Both are zero-cost (compile-time) when neither env var is set.
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>

/* ── Cached trace level checks ────────────────────────────────────────── */

static inline int strace_level(void) {
    static int cached = -1;   /* -1 = need to check */
    if (cached == -1) {
        const char* e = getenv("STRACE");
        cached = e ? atoi(e) : 0;
    }
    return cached;
}

static inline int perf_trace_level(void) {
    static int cached = -1;   /* -1 = need to check */
    if (cached == -1) {
        const char* e = getenv("PERF_TRACE");
        cached = e ? atoi(e) : 0;
    }
    return cached;
}

/* ── Debug breadcrumb macros (STRACE) ─────────────────────────────────── */
/* These can be granular — per-layer, per-head, per-token. Acceptable perf cost. */

#define STRACE(comp, evt, fmt, ...)                                          \
    do {                                                                     \
        if (strace_level() >= 1)                                             \
            fprintf(stderr, "BREADCRUMB comp=%s evt=%s " fmt "\n",           \
                    comp, evt, ##__VA_ARGS__);                               \
    } while (0)

/* Convenience wrappers for common ligguf phases */
#define STRACE_MODEL(arch)        STRACE("model", "load", "arch=%s", arch)
#define STRACE_TOKENIZER(vocab)   STRACE("tokenizer", "loaded", "vocab=%d", vocab)
#define STRACE_INFER(layer)       STRACE("inference", "layer", "layer=%d", layer)
#define STRACE_ATTN(layer, head)  STRACE("attention", "layer", "layer=%d head=%d", layer, head)
#define STRACE_FFN(layer)         STRACE("ffn", "layer", "layer=%d", layer)
#define STRACE_SAMPLE(tok)        STRACE("sampling", "token", "tok=%d", tok)
#define STRACE_KVCACHE(pos)       STRACE("kvcache", "pos", "pos=%d", pos)
#define STRACE_DEQUANT(t, n)      STRACE("dequant", "row", "type=%d n=%I64d", t, (int64_t)(n))
#define STRACE_MATMUL(name, n, d) STRACE("matmul", name, "n=%d d=%d", n, d)

/* ── Lightweight performance tracing (PERF_TRACE) ─────────────────────── */
/* ONLY at phase boundaries — never inside tight loops. */

#ifdef __cplusplus
#include <chrono>

struct perf_scope {
    const char* comp;
    const char* evt;
    int lvl;
    std::chrono::steady_clock::time_point t0;
    perf_scope(const char* c, const char* e)
        : comp(c), evt(e),
          lvl(perf_trace_level()),
          t0(std::chrono::steady_clock::now())
    {
        if (lvl >= 1)
            fprintf(stderr, "PERF_SCOPE comp=%s evt=%s begin\n", comp, evt);
    }
    ~perf_scope() {
        if (lvl >= 1) {
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            fprintf(stderr, "PERF_SCOPE comp=%s evt=%s end ms=%.3f\n",
                    comp, evt, ms);
        }
    }
};

/* MSVC preprocessor: need two-level indirection for __COUNTER__ expansion before ## */
#define PERF_SCOPE_CONCAT_(a, b) a##b
#define PERF_SCOPE_CONCAT(a, b) PERF_SCOPE_CONCAT_(a, b)
#define PERF_SCOPE(comp, evt) PERF_SCOPE_CONCAT(perf_scope _ps_, __COUNTER__)(comp, evt)

#else  /* C mode — no chrono, just conditional print at begin */

#define PERF_SCOPE(comp, evt)                                      \
    do {                                                          \
        if (perf_trace_level() >= 1)                              \
            fprintf(stderr, "PERF_SCOPE comp=%s evt=%s begin\n",  \
                    comp, evt);                                   \
    } while (0)

#endif  /* __cplusplus */

/* Phase-level perf scopes — use sparingly, only at high boundaries */
#define PERF_INFERENCE()      PERF_SCOPE("inference", "token")
#define PERF_MODEL_LOAD()     PERF_SCOPE("model", "load")
#define PERF_MODEL_LOAD_END() PERF_SCOPE("model", "load_end")
#define PERF_TOKENIZE()       PERF_SCOPE("tokenizer", "tokenize")
#define PERF_TOKENIZE_END()   PERF_SCOPE("tokenizer", "tokenize_end")
#define PERF_GENERATE()       PERF_SCOPE("generate", "loop")
#define PERF_SAMPLE()         PERF_SCOPE("sampling", "token")
#define PERF_CHAT_TURN()      PERF_SCOPE("chat", "turn")
