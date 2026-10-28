// AVX2 compute (L1-resident) + streaming bandwidth probes
#include <immintrin.h>
#include <stdio.h>
#include <chrono>
#include <cstdint>

__declspec(align(32)) static float a[1024];
__declspec(align(32)) static float b[1024];
__declspec(align(32)) static float big_a[1<<22];
__declspec(align(32)) static float big_b[1<<22];

int main()
{
    const int M = 1024;
    for (int i=0;i<M;i++){ a[i]=1.0f; b[i]=2.0f; }
    __m256 acc0=_mm256_setzero_ps(),acc1=_mm256_setzero_ps(),acc2=_mm256_setzero_ps(),acc3=_mm256_setzero_ps();
    long reps = 4000000;
    auto t0=std::chrono::steady_clock::now();
    for (long r=0;r<reps;r++){
        for (int i=0;i<M;i+=32){
            acc0=_mm256_fmadd_ps(_mm256_load_ps(a+i   ),_mm256_load_ps(b+i   ),acc0);
            acc1=_mm256_fmadd_ps(_mm256_load_ps(a+i+8 ),_mm256_load_ps(b+i+8 ),acc1);
            acc2=_mm256_fmadd_ps(_mm256_load_ps(a+i+16),_mm256_load_ps(b+i+16),acc2);
            acc3=_mm256_fmadd_ps(_mm256_load_ps(a+i+24),_mm256_load_ps(b+i+24),acc3);
        }
    }
    auto t1=std::chrono::steady_clock::now();
    double s=std::chrono::duration<double>(t1-t0).count();
    double mac=(double)reps*M;
    printf("[compute L1] elapsed=%.3f s  GFLOP/s=%.2f  (sink=%f)\n",s, mac*2/s/1e9,
           _mm256_cvtss_f32(_mm256_add_ps(_mm256_add_ps(acc0,acc1),_mm256_add_ps(acc2,acc3))));

    // streaming read bandwidth
    const int N = 1<<22;
    for (int i=0;i<N;i++){ big_a[i]=1.0f; big_b[i]=2.0f; }
    __m256 s0=_mm256_setzero_ps();
    long reps2 = 20;
    auto u0=std::chrono::steady_clock::now();
    for (long r=0;r<reps2;r++){
        for (int i=0;i<N;i+=8)
            s0=_mm256_add_ps(s0,_mm256_add_ps(_mm256_load_ps(big_a+i),_mm256_load_ps(big_b+i)));
    }
    auto u1=std::chrono::steady_clock::now();
    double s2=std::chrono::duration<double>(u1-u0).count();
    double bytes=(double)reps2*N*4*2;
    printf("[stream] elapsed=%.3f s  GB/s=%.2f  (sink=%f)\n",s2, bytes/s2/1e9, _mm256_cvtss_f32(s0));
    return 0;
}
