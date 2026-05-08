@echo off
REM ============================================================
REM  Step 3: coa-compiler  model/resnet18.mlir -> output/resnet18.vliw
REM ============================================================
REM  前提：coa-compiler 已构建（见 compiler/build.bat）
REM ============================================================

SET ROOT=%~dp0..\..
SET COA_COMPILER=%ROOT%\compiler\build\bin\Release\coa-compiler.exe
SET INPUT=%~dp0model\resnet18.mlir
SET OUTPUT_DIR=%~dp0output
SET OUTPUT=%OUTPUT_DIR%\resnet18.vliw

REM ---- Sanity checks ----
IF NOT EXIST "%COA_COMPILER%" (
    echo [Step3] ERROR: coa-compiler not found at %COA_COMPILER%
    echo         Run  compiler\build.bat  first.
    exit /b 1
)
IF NOT EXIST "%INPUT%" (
    echo [Step3] ERROR: %INPUT% not found.
    echo         Run 02_import_mlir.py first.
    exit /b 1
)

mkdir "%OUTPUT_DIR%" 2>nul

echo [Step3] Compiling %INPUT% ...
echo.

"%COA_COMPILER%" ^
    --output "%OUTPUT%" ^
    --weight-base 0x08000000 ^
    --bias-base   0xC0000000 ^
    --act-base    0x10000000 ^
    "%INPUT%"

IF ERRORLEVEL 1 (
    echo.
    echo [Step3] FAILED.
    exit /b 1
)

echo.
echo [Step3] Done -^> %OUTPUT%
echo [Step3] Next: run  04_verify.py

REM ---- Optional: also emit the lowered MLIR for inspection ----
SET LOWERED=%OUTPUT_DIR%\resnet18_lowered.mlir
SET COA_OPT=%ROOT%\compiler\build\bin\Release\coa-opt.exe

IF EXIST "%COA_OPT%" (
    echo.
    echo [Step3] Emitting lowered MLIR for inspection...
    "%COA_OPT%" ^
        --coa-shape-infer ^
        --coa-op-fusion ^
        --coa-tiling ^
        --coa-addr-assign ^
        --coa-legalize ^
        "%INPUT%" -o "%LOWERED%"
    echo [Step3] Lowered MLIR -^> %LOWERED%
)
