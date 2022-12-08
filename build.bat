@echo off

cd %~dp0

if exist build (
    rmdir /s /q build
)

mkdir build
cd build

cmake ..
cmake --build . --config Release

cd ..
if not exist build\Release\lbmcl.exe (
    echo Build failed
    rmdir /s /q build
    exit /b 1
)

copy src\lbm.cl build\Release
copy src\gl_shader\vertex.vert build\Release
copy src\gl_shader\render.frag build\Release
copy res\mask.jpg build\Release

echo Build succeeded
