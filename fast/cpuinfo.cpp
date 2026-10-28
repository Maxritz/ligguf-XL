// Local CPU feature probe (CPUID). Reports what the inference kernels may use.
#include <intrin.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static void cpuid(int leaf, int sub, int r[4]) { __cpuidex(r, leaf, sub); }
static uint64_t xgetbv0(void) { return _xgetbv(0); }

int main()
{
    int r[4];

    cpuid(0, 0, r);
    char vendor[13]; memcpy(vendor, &r[1], 4); memcpy(vendor + 4, &r[3], 4); memcpy(vendor + 8, &r[2], 4);
    vendor[12] = 0;
    printf("vendor          : %s\n", vendor);
    int maxleaf = r[0];

    if (maxleaf >= 0x80000004) {
        char brand[49];
        for (int i = 0; i < 3; i++) { cpuid(0x80000002 + i, 0, r); memcpy(brand + i * 16, r, 16); }
        brand[48] = 0;
        printf("brand           : %s\n", brand);
    }

    cpuid(1, 0, r);
    const bool sse42 = r[2] & (1 << 20);
    const bool osxsave = r[2] & (1 << 27);
    const bool avx = r[2] & (1 << 28);
    const bool fma = r[2] & (1 << 12);
    const bool f16c = r[2] & (1 << 29);
    const bool aes = r[2] & (1 << 25);

    uint64_t xcr0 = osxsave ? xgetbv0() : 0;
    const bool ymm_state = (xcr0 & 0x6) == 0x6;            // XMM + YMM saved by OS
    const bool zmm_state = (xcr0 & 0xE6) == 0xE6;          // + opmask/ZMM/Hi16

    cpuid(7, 0, r);
    const int l7e = r[0], l7b = r[1], l7c = r[2], l7d = r[3];
    const bool avx2 = l7b & (1 << 5);
    const bool bmi1 = l7b & (1 << 3);
    const bool bmi2 = l7b & (1 << 8);
    const bool fma3 = l7b & (1 << 16);       // AVX512F
    const bool avx512f = l7b & (1 << 16);
    const bool avx512bw = l7b & (1 << 30);
    const bool avx512vl = l7b & (1 << 31);
    const bool avx512vnni = l7c & (1 << 11);
    const bool avx512bf16 = l7e & (1 << 5);

    cpuid(7, 1, r);
    const bool avxvnni = r[0] & (1 << 4);
    const bool avx512vbmi2 = r[2] & (1 << 6);
    const bool avxvnni_int8 = r[3] & (1 << 4);
    const bool avxvnni_int16 = r[3] & (1 << 10);
    const bool prefetchi = r[3] & (1 << 14);
    const bool avx_ifma = (r[3] & (1 << 21)) != 0;
    const bool avx512_vp2intersect = r[0] & (1 << 6);

    int l2[4];
    cpuid(0x80000008, 0, l2);
    printf("max leaf        : 0x%x\n", maxleaf);
    printf("\n-- baseline --\n");
    printf("SSE4.2          : %d\n", sse42);
    printf("FMA3            : %d\n", fma);
    printf("F16C            : %d\n", f16c);
    printf("AES-NI          : %d\n", aes);
    printf("AVX (OS state)  : %d  (XCR0=0x%llx YMM=%d ZMM=%d)\n", avx && ymm_state, (unsigned long long)xcr0, ymm_state, zmm_state);
    printf("AVX2            : %d\n", avx2 && ymm_state);
    printf("BMI1 / BMI2     : %d / %d\n", bmi1, bmi2);

    printf("\n-- the interesting part for this engine --\n");
    printf("AVX-VNNI        : %d  (vpdpbusd, 256-bit int8 dot)\n", avxvnni && ymm_state);
    printf("AVX-VNNI-INT8   : %d  (vpdpbusds 16-bit saturating)\n", avxvnni_int8);
    printf("AVX-VNNI-INT16  : %d\n", avxvnni_int16);
    printf("AVX512F         : %d  (OS state: %d)\n", avx512f, zmm_state);
    printf("AVX512BW/VL     : %d / %d\n", avx512bw, avx512vl);
    printf("AVX512-VNNI     : %d\n", avx512vnni);
    printf("AVX512-BF16     : %d\n", avx512bf16);
    printf("AVX-IFMA        : %d\n", avx_ifma);
    printf("PREFETCHI       : %d\n", prefetchi);
    printf("AVX512_VP2INT   : %d\n", avx512_vp2intersect);
    printf("AVX512VBMI2     : %d\n", avx512vbmi2);
    return 0;
}
