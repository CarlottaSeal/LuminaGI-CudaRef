@echo off
setlocal
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "MSVC_ROOT=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130"
set "SDK_ROOT=C:\Program Files (x86)\Windows Kits\10"
set "SDK_VER=10.0.22621.0"
set "PATH=%CUDA_PATH%\bin;%MSVC_ROOT%\bin\Hostx64\x64;%SDK_ROOT%\bin\%SDK_VER%\x64;%PATH%"
set "INCLUDE=%MSVC_ROOT%\include;%SDK_ROOT%\Include\%SDK_VER%\ucrt;%SDK_ROOT%\Include\%SDK_VER%\um;%SDK_ROOT%\Include\%SDK_VER%\shared"
set "LIB=%MSVC_ROOT%\lib\x64;%SDK_ROOT%\Lib\%SDK_VER%\ucrt\x64;%SDK_ROOT%\Lib\%SDK_VER%\um\x64"

set LB=%1
if "%LB%"=="" set LB=3

echo === launch_bounds(256, %LB%) ===
"%CUDA_PATH%\bin\nvcc.exe" -arch=sm_89 -O2 -std=c++17 --allow-unsupported-compiler ^
    -DACC_LB_BLOCKS=%LB% -Xptxas=-v ^
    -I include -I third_party ^
    src\main.cpp src\pathtracer.cu src\scene_loader.cpp src\bvh.cpp ^
    -o build\bin\cuda_ref.exe 2>&1 | findstr /C:"accumulate_kernel" /C:"Used"

echo.
echo --- 3 timing runs ---
for /L %%i in (1,1,3) do (
    build\bin\cuda_ref.exe "C:\p4\Personal\SD\LuminaGI\Run\Screenshots\manual_20260417_011144.json" -o output\occ_%LB%.png --spp 64 --bounces 2 2>nul | findstr /C:"kernel:"
)
