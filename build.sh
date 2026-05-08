#!/usr/bin/env bash
# ============================================================
#  fpga-mlir Build Script (Linux)
# ============================================================
#
# Prerequisites:
#   Build llvm-project with MLIR enabled (one-time setup):
#
#     cmake -S llvm-project/llvm -B llvm-project/build \
#           -G Ninja \
#           -DLLVM_ENABLE_PROJECTS="mlir;clang" \
#           -DLLVM_TARGETS_TO_BUILD="X86" \
#           -DCMAKE_BUILD_TYPE=Release \
#           -DLLVM_ENABLE_ASSERTIONS=ON
#     cmake --build llvm-project/build -j$(nproc)
#
# Usage:
#   ./build.sh                  # configure + build (Release)
#   ./build.sh clean            # wipe build dir first, then build
#   ./build.sh --debug          # Debug build
#   MLIR_DIR=/custom/path ./build.sh
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT}/build"
BUILD_TYPE="Release"

# Parse args
CLEAN=false
for arg in "$@"; do
    case "${arg}" in
        clean)   CLEAN=true ;;
        --debug) BUILD_TYPE="Debug" ;;
        *) echo "[build.sh] Unknown argument: ${arg}"; exit 1 ;;
    esac
done

# ---- Locate MLIR installation ----
# Default: sibling llvm-project build. Override via MLIR_DIR env var.
MLIR_DIR="${MLIR_DIR:-${ROOT}/../llvm-project/build/lib/cmake/mlir}"

if [[ ! -f "${MLIR_DIR}/MLIRConfig.cmake" ]]; then
    echo "[ERROR] MLIRConfig.cmake not found at: ${MLIR_DIR}"
    echo "        Build llvm-project first, or set:"
    echo "          MLIR_DIR=<llvm_build>/lib/cmake/mlir ./build.sh"
    exit 1
fi

# ---- Optional clean ----
if [[ "${CLEAN}" == true ]]; then
    echo "[build.sh] Cleaning ${BUILD_DIR} ..."
    rm -rf "${BUILD_DIR}"
fi

# ---- Configure ----
echo "[build.sh] Configuring (MLIR_DIR=${MLIR_DIR}, type=${BUILD_TYPE}) ..."
cmake -S "${ROOT}" -B "${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DMLIR_DIR="${MLIR_DIR}"

# ---- Build ----
echo "[build.sh] Building ..."
cmake --build "${BUILD_DIR}" --target coa-opt coa-compiler -j"$(nproc)"

echo ""
echo "[build.sh] Build succeeded!"
echo "  coa-opt:       ${BUILD_DIR}/bin/coa-opt"
echo "  coa-compiler:  ${BUILD_DIR}/bin/coa-compiler"
echo ""
echo "Quick-start:"
echo "  ${BUILD_DIR}/bin/coa-opt \\"
echo "      --coa-shape-infer --coa-tiling --coa-addr-assign \\"
echo "      --coa-legalize model.mlir"
echo ""
echo "  ${BUILD_DIR}/bin/coa-compiler --output model.vliw model.mlir"
echo ""
echo "Run tests:"
echo "  ctest --test-dir ${BUILD_DIR} -V --label-regex python"
