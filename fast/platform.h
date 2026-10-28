/* LiGGUF - Windows platform compatibility layer
 *
 * Replaces POSIX mmap/fstat/open/close/getopt with Windows equivalents
 * so the FAST engine compiles and runs under MSVC on Windows.
 */

#pragma once

#ifdef _WIN32

#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

/* mmap emulation on Windows */
#ifndef MAP_SHARED
#define MAP_SHARED 0x01
#endif
#ifndef PROT_READ
#define PROT_READ  0x01
#endif
#ifndef PROT_WRITE
#define PROT_WRITE 0x02
#endif

#ifndef MAP_FAILED
#define MAP_FAILED ((void*)-1)
#endif

typedef void* mmap_t;

inline void* mmap_wrap(HANDLE hFile, uint64_t /*fileSize*/)
{
    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) return MAP_FAILED;
    void* p = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);   /* MapViewOfFile keeps the mapping alive */
    return p;
}

inline int munmap_wrap(void* addr, uint64_t /*len*/)
{
    if (addr == MAP_FAILED || addr == nullptr) return -1;
    return UnmapViewOfFile(addr) ? 0 : -1;
}

/* open() emulation */
inline HANDLE open_wrap(const char* path)
{
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    return h == INVALID_HANDLE_VALUE ? (HANDLE)-1 : h;
}

inline int close_wrap(HANDLE h)
{
    if (h == (HANDLE)-1 || h == NULL) return -1;
    return CloseHandle(h) ? 0 : -1;
}

/* fstat emulation */
inline uint64_t file_size_wrap(HANDLE h)
{
    LARGE_INTEGER li;
    GetFileSizeEx(h, &li);
    return (uint64_t)li.QuadPart;
}

/* stat emulation for open_mmap path */
struct stat_wrap {
    uint64_t st_size;
};
inline int fstat_wrap(HANDLE h, stat_wrap* st)
{
    st->st_size = file_size_wrap(h);
    return 0;
}

/* localtime_r wrapper — Windows provides localtime_s with reversed args */
#ifdef _WIN32
inline struct tm* localtime_r(const time_t* t, struct tm* tm) {
    return localtime_s(tm, t) == 0 ? tm : nullptr;
}
inline void usleep_wrap(uint64_t us) {
    Sleep((DWORD)(us / 1000));
}
#endif

/* ── Minimal getopt replacement for Windows ──────────────────────────── */

#ifdef __cplusplus
inline
#else
extern
#endif
char* optarg = nullptr;

#ifdef __cplusplus
inline
#else
extern
#endif
int optind = 0;

inline int getopt(int argc, char* argv[], const char* optstring)
{
    static char* next = nullptr;
    static char* arg = nullptr;

    if (optind == 0) {
        optind = 1;
        next = nullptr;
        arg = nullptr;
    }

    if (next == nullptr || *next == '\0') {
        if (optind >= argc) return -1;
        arg = argv[optind];
        if (arg[0] != '-' || arg[1] == '\0') return -1;
        if (arg[1] == '-' && arg[2] == '\0') { optind++; return -1; }
        next = arg + 1;
    }

    char c = *next++;
    char* spec = (char*)strchr(optstring, c);
    if (spec == nullptr) {
        fprintf(stderr, "Unknown option -%c\n", c);
        return '?';
    }
    optarg = nullptr;
    if (spec[1] == ':') {
        if (*next != '\0') {
            optarg = next;
            next = nullptr;
        } else {
            if (optind + 1 >= argc) return '?';
            optarg = argv[++optind];
        }
    }
    int result = c;
    if (next == nullptr || *next == '\0')
        optind++;
    return result;
}

#else   /* !_WIN32 — native POSIX */

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>

#define close_wrap close
#define open_wrap  open
#define fstat_wrap fstat
#define file_size_wrap(s) ((uint64_t)(s).st_size)
#define usleep_wrap(us) usleep(us)

#endif  /* _WIN32 */
