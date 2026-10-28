@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul 2>&1
cl /nologo /std:c++17 /O2 /arch:AVX2 /Fe:bench_flops.exe bench_flops.cpp
