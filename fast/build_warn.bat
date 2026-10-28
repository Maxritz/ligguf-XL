@echo off
REM compile every engine TU with /W4 and report warnings (no link)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul 2>&1
cl /nologo /std:c++17 /EHsc /O2 /arch:AVX2 /openmp /W4 /D_CRT_SECURE_NO_WARNINGS /D__AVX2__ /c ^
   lil_math.cpp lil_quants.cpp hadamard.cpp lil_gguf.cpp common.cpp tokenize.cpp vdb.cpp ligguf_fast_qk.cpp
echo ---- done ----
