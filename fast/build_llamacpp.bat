@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul 2>&1
cd /d C:\Users\rina0423\Desktop\experiment\llama.cpp
cmake --build build-win --config Release --target llama-completion llama-tokenize -j 8
