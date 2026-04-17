@echo off
setlocal

set "MSVC_ROOT=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130"
set "SDK_ROOT=C:\Program Files (x86)\Windows Kits\10"
set "SDK_VER=10.0.26100.0"
set "PATH=%MSVC_ROOT%\bin\Hostx64\x64;%SDK_ROOT%\bin\%SDK_VER%\x64;%PATH%"
set "INCLUDE=%MSVC_ROOT%\include;%SDK_ROOT%\Include\%SDK_VER%\ucrt;%SDK_ROOT%\Include\%SDK_VER%\um;%SDK_ROOT%\Include\%SDK_VER%\shared"
set "LIB=%MSVC_ROOT%\lib\x64;%SDK_ROOT%\Lib\%SDK_VER%\ucrt\x64;%SDK_ROOT%\Lib\%SDK_VER%\um\x64"

if not exist build\bin mkdir build\bin

cl /nologo /std:c++17 /EHsc /O2 /W3 ^
    /I include /I third_party ^
    src\test_bvh.cpp src\bvh.cpp src\scene_loader.cpp ^
    /Fe:build\bin\test_bvh.exe ^
    /Fo:build\bin\
if errorlevel 1 (echo BUILD_FAILED & exit /b 1)
echo BUILD_OK
