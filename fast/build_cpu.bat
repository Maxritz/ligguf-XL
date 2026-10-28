@echo off
REM build_cpu.bat [arch] [extra-defines]
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul 2>&1
set ARCH=%1
if "%ARCH%"=="" set ARCH=AVX2
cl /nologo /std:c++17 /O2 /arch:%ARCH% /DHAVE_VNNI /Fe:bench_cpu.exe bench_cpu.cpp
