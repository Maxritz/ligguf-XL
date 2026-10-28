#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>

// GGUF value types (correct from spec)
#define GGUF_UINT8    0
#define GGUF_INT8     1
#define GGUF_UINT16   2
#define GGUF_INT16    3
#define GGUF_UINT32   4
#define GGUF_INT32    5
#define GGUF_FLOAT32  6
#define GGUF_BOOL     7
#define GGUF_STRING   8
#define GGUF_ARRAY    9
#define GGUF_UINT64   10
#define GGUF_INT64    11
#define GGUF_FLOAT64  12

int main(int argc, char** argv) {
    if (argc < 2) { printf("Usage: dump_gguf <model.gguf>\n"); return 1; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { printf("Cannot open %s\n", argv[1]); return 1; }

    // Read header: magic(4) + version(4) + tensor_count(8) + kv_count(8)
    char magic[4];
    fread(magic, 1, 4, f);
    uint32_t ver;
    fread(&ver, 4, 1, f);
    uint64_t nten;
    fread(&nten, 8, 1, f);
    uint64_t kv_count;
    fread(&kv_count, 8, 1, f);
    
    printf("Version: %u, Tensors: %llu, KV pairs: %llu\n", ver, 
           (unsigned long long)nten, (unsigned long long)kv_count);
    fflush(stdout);

    // Read metadata KV pairs
    for (uint64_t i = 0; i < kv_count; i++) {
        uint64_t k_len;
        fread(&k_len, 8, 1, f);
        char key[512] = {0};
        fread(key, 1, k_len, f);

        uint32_t val_type;
        fread(&val_type, 4, 1, f);

        if (val_type == GGUF_STRING) {
            uint64_t s_len;
            fread(&s_len, 8, 1, f);
            char* s_val = new char[s_len+1];
            fread(s_val, 1, s_len, f);
            s_val[s_len] = 0;
            printf("KV: %s = %s\n", key, s_val);
            fflush(stdout);
            delete[] s_val;
        } else if (val_type == GGUF_UINT32 || val_type == GGUF_INT32) {
            uint32_t v; fread(&v, 4, 1, f);
            printf("KV: %s = %u\n", key, v);
            fflush(stdout);
        } else if (val_type == GGUF_FLOAT32) {
            float v; fread(&v, 4, 1, f);
            printf("KV: %s = %f\n", key, v);
            fflush(stdout);
        } else if (val_type == GGUF_UINT64 || val_type == GGUF_INT64) {
            uint64_t v; fread(&v, 8, 1, f);
            printf("KV: %s = %llu\n", key, (unsigned long long)v);
            fflush(stdout);
        } else if (val_type == GGUF_BOOL) {
            uint8_t v; fread(&v, 1, 1, f);
            printf("KV: %s = %u\n", key, v);
        } else if (val_type == GGUF_UINT8 || val_type == GGUF_INT8) {
            uint8_t v; fread(&v, 1, 1, f);
        } else if (val_type == GGUF_UINT16 || val_type == GGUF_INT16) {
            uint16_t v; fread(&v, 2, 1, f);
        } else if (val_type == GGUF_FLOAT64) {
            double v; fread(&v, 8, 1, f);
        } else if (val_type == GGUF_ARRAY) {
            uint32_t arr_type; fread(&arr_type, 4, 1, f);
            uint64_t arr_len; fread(&arr_len, 8, 1, f);
            printf("KV: %s = [ARRAY type=%u len=%llu]\n", key, arr_type, (unsigned long long)arr_len);
            fflush(stdout);
            // Skip array elements
            for (uint64_t a = 0; a < arr_len; a++) {
                if (arr_type == GGUF_STRING) {
                    uint64_t s_len; fread(&s_len, 8, 1, f);
                    fseek(f, s_len, SEEK_CUR);
                } else if (arr_type <= 6 || arr_type == 10 || arr_type == 11 || arr_type == 12) {
                    uint64_t sizes[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
                    sizes[GGUF_UINT8] = 1; sizes[GGUF_INT8] = 1;
                    sizes[GGUF_UINT16] = 2; sizes[GGUF_INT16] = 2;
                    sizes[GGUF_UINT32] = 4; sizes[GGUF_INT32] = 4; sizes[GGUF_FLOAT32] = 4;
                    sizes[GGUF_BOOL] = 1;
                    sizes[GGUF_UINT64] = 8; sizes[GGUF_INT64] = 8; sizes[GGUF_FLOAT64] = 8;
                    fseek(f, sizes[arr_type], SEEK_CUR);
                }
            }
        } else {
            printf("KV: %s (unknown type %u)\n", key, val_type);
            fflush(stdout);
        }
    }

    printf("\n--- TENSORS ---\n");
    fflush(stdout);
    for (uint64_t i = 0; i < nten; i++) {
        uint64_t name_len;
        fread(&name_len, 8, 1, f);
        char name[512] = {0};
        fread(name, 1, name_len, f);

        uint32_t n_dim;
        fread(&n_dim, 4, 1, f);
        uint64_t dims[4] = {0,0,0,0};
        for (uint32_t d = 0; d < n_dim && d < 4; d++) fread(&dims[d], 8, 1, f);

        uint32_t dtype;
        fread(&dtype, 4, 1, f);
        uint64_t offset;
        fread(&offset, 8, 1, f);

        printf("%-75s type=%u dims=[", name, dtype);
        for (uint32_t d = 0; d < n_dim; d++) {
            printf("%llu", (unsigned long long)dims[d]);
            if (d < n_dim-1) printf(",");
        }
        printf("]\n");
        fflush(stdout);
    }

    fclose(f);
    return 0;
}
