@echo off
REM LiGGUF FAST engine Windows build script for MSVC (VS2022 BuildTools)
REM Usage: build_win.bat [debug]

setlocal

REM --- Locate VS2022 BuildTools ---
if not exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" (
    echo Error: VS2022 BuildTools not found. Install from:
    echo   https://visualstudio.microsoft.com/visual-cpp-build-tools/
    exit /b 1
)

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64

set OPT=/Od
set DEFS=/Zi
if /i not "%1"=="debug" (
    set OPT=/O2
    set DEFS=
)

del *.obj 2>nul

cl.exe /std:c++17 /EHsc %OPT% /arch:AVX2 /openmp /D_CRT_SECURE_NO_WARNINGS /D__AVX2__ %DEFS% ^
    lil_math.cpp ^
    lil_quants.cpp ^
    hadamard.cpp ^
    lil_gguf.cpp ^
    common.cpp ^
    tokenize.cpp ^
    vdb.cpp ^
    ligguf_fast_qk.cpp ^
    /Fe:ligguf-fast.exe

endlocal
