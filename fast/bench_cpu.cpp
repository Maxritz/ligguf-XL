// Local CPU ceiling + AVX2 vs AVX-VNNI int8 dot comparison.
#include <immintrin.h>
#include <stdio.h>
#include <chrono>
#include <cstdint>

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

__declspec(align(64)) static int8_t qa[1 << 16];
__declspec(align(64)) static int8_t qb[1 << 16];
__declspec(align(64)) static float  fa[1 << 10];
__declspec(align(64)) static float  fb[1 << 10];

static volatile float g_sink;

// ---- effective core clock: serial dependent FMA chain (latency bound) --------
static void measure_clock() {
    __m256 x = _mm256_set1_ps(1.0000001f);
    const __m256 a = _mm256_set1_ps(1.0000001f);
    const __m256 b = _mm256_set1_ps(0.0000001f);
    const long N = 300000000L;
    double t0 = now();
    for (long i = 0; i < N; i++) x = _mm256_fmadd_ps(x, a, b);
    double t1 = now();
    g_sink = _mm256_cvtss_f32(x);
    // FMA latency on Golden Cove / Redwood Cove is 4 cycles
    double ghz = (double)N * 4.0 / (t1 - t0) / 1e9;
    printf("effective clock (~4cyc FMA latency) : %.2f GHz\n", ghz);
}

// ---- AVX2 FMA peak ----------------------------------------------------------
static void measure_fma() {
    __m256 a0 = _mm256_set1_ps(1.000001f), a1 = a0, a2 = a0, a3 = a0;
    __m256 a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    const __m256 m = _mm256_set1_ps(0.999999f);
    const long N = 200000000L;
    double t0 = now();
    for (long i = 0; i < N; i++) {
        a0 = _mm256_fmadd_ps(a0, m, m);
        a1 = _mm256_fmadd_ps(a1, m, m);
        a2 = _mm256_fmadd_ps(a2, m, m);
        a3 = _mm256_fmadd_ps(a3, m, m);
        a4 = _mm256_fmadd_ps(a4, m, m);
        a5 = _mm256_fmadd_ps(a5, m, m);
        a6 = _mm256_fmadd_ps(a6, m, m);
        a7 = _mm256_fmadd_ps(a7, m, m);
    }
    double t1 = now();
    __m256 s0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m256 s1 = _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7));
    g_sink = _mm256_cvtss_f32(_mm256_add_ps(s0, s1));
    double fma = (double)N * 8.0;
    printf("AVX2 FMA peak (8 chains, 2 flop/fma): %.1f GFLOP/s\n", fma * 2.0 * 8.0 / (t1 - t0) / 1e9);
}

// ---- int8 dot: current engine path (maddubs + madd) -------------------------
static inline __m256i mul_sum_i8_pairs(__m256i x, __m256i y) {
    __m256i ax = _mm256_sign_epi8(x, x);
    __m256i sy = _mm256_sign_epi8(y, x);
    __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return _mm256_madd_epi16(_mm256_set1_epi16(1), dot);
}

static void measure_dot(const char* label, int use_vnni) {
    const int bytes = 1 << 16;
    const int iters = 40000;
    __m256i acc = _mm256_setzero_si256();
    double t0 = now();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < bytes; i += 32) {
            __m256i x = _mm256_load_si256((const __m256i*)(qa + i));
            __m256i y = _mm256_load_si256((const __m256i*)(qb + i));
            if (use_vnni) {
                __m256i ax = _mm256_sign_epi8(x, x);
                __m256i sy = _mm256_sign_epi8(y, x);
                acc = _mm256_dpbusd_epi32(acc, ax, sy);
            } else {
                acc = _mm256_add_epi32(acc, mul_sum_i8_pairs(x, y));
            }
        }
    }
    double t1 = now();
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    g_sink = (float)_mm_cvtsi128_si32(s);
    double macs = (double)iters * bytes;   // 1 MAC per byte pair
    printf("%-36s: %7.1f GMAC/s  (%5.2f GB/s of int8)\n", label, macs / (t1 - t0) / 1e9, macs / (t1 - t0) / 1e9);
}

// ---- streaming bandwidth ----------------------------------------------------
__declspec(align(64)) static float ba[1 << 22];
__declspec(align(64)) static float bb[1 << 22];
static void measure_stream() {
    const int n = 1 << 22;
    __m256 s = _mm256_setzero_ps();
    const int reps = 25;
    double t0 = now();
    for (int r = 0; r < reps; r++)
        for (int i = 0; i < n; i += 8)
            s = _mm256_add_ps(s, _mm256_add_ps(_mm256_load_ps(ba + i), _mm256_load_ps(bb + i)));
    double t1 = now();
    g_sink = _mm256_cvtss_f32(s);
    printf("stream read bandwidth               : %7.1f GB/s\n",
           (double)reps * n * 4 * 2 / (t1 - t0) / 1e9);
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 0; i < (1 << 16); i++) qa[i] = (int8_t)((i * 37) & 0x7f), qb[i] = (int8_t)((i * 91) & 0x7f);
    for (int i = 0; i < (1 << 10); i++) fa[i] = fb[i] = 1.0f;
    for (int i = 0; i < (1 << 22); i++) ba[i] = 1.0f;

    measure_clock();
    measure_fma();
    printf("\nint8 dot, 32 lanes per step:\n");
    measure_dot("current path (maddubs+madd)", 0);
#ifdef HAVE_VNNI
    measure_dot("AVX-VNNI (vpdpbusd)", 1);
#endif
    printf("\n");
    measure_stream();
    return 0;
}
