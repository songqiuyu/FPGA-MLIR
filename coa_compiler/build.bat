@echo off
REM ============================================================
REM  COA Compiler Build Script (Windows / MSVC or Clang-CL)
REM ============================================================
REM
REM Prerequisites:
REM   1. Build llvm-project with MLIR enabled:
REM        cmake -S llvm-project\llvm -B llvm-project\build ^
REM              -G "Visual Studio 17 2022" -A x64 ^
REM              -DLLVM_ENABLE_PROJECTS="mlir;clang" ^
REM              -DLLVM_TARGETS_TO_BUILD="X86" ^
REM              -DCMAKE_BUILD_TYPE=Release ^
REM              -DLLVM_ENABLE_ASSERTIONS=ON
REM        cmake --build llvm-project\build --config Release -j8
REM
REM   2. Set MLIR_DIR below to <llvm_build>/lib/cmake/mlir
REM
REM Usage:
REM   cd coa_compiler
REM   build.bat [clean]
REM ============================================================

SET ROOT=%~dp0..
SET BUILD_DIR=%~dp0build

REM ---- Locate MLIR installation ----
SET MLIR_DIR=%ROOT%\llvm-project\build\lib\cmake\mlir
IF NOT EXIST "%MLIR_DIR%\MLIRConfig.cmake" (
    echo [ERROR] MLIRConfig.cmake not found at %MLIR_DIR%
    echo         Please build llvm-project first, or update MLIR_DIR in build.bat
    exit /b 1
)

REM ---- Optional clean ----
IF "%1"=="clean" (
    echo [build.bat] Cleaning build directory...
    rmdir /s /q "%BUILD_DIR%" 2>nul
)

REM ---- Configure ----
echo [build.bat] Configuring COA compiler...
cmake -S "%~dp0" -B "%BUILD_DIR%" ^
    -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DMLIR_DIR="%MLIR_DIR%"

IF ERRORLEVEL 1 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

REM ---- Build ----
echo [build.bat] Building...
cmake --build "%BUILD_DIR%" --config Release --target coa-opt coa-compiler -j8

IF ERRORLEVEL 1 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo [build.bat] Build succeeded!
echo   coa-opt:       %BUILD_DIR%\bin\Release\coa-opt.exe
echo   coa-compiler:  %BUILD_DIR%\bin\Release\coa-compiler.exe
echo.
echo Example usage:
echo   %BUILD_DIR%\bin\Release\coa-opt.exe ^
echo       --coa-shape-infer --coa-tiling --coa-addr-assign ^
echo       --coa-legalize model.mlir
echo.
echo   %BUILD_DIR%\bin\Release\coa-compiler.exe ^
echo       --output model.vliw model.mlir
