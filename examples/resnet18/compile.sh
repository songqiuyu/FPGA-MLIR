#!/usr/bin/env bash
# ============================================================
#  Step 3: coa-compiler  model/resnet18.mlir -> output/resnet18.vliw
# ============================================================
#  Prerequisite: build coa-compiler first:
#    ./build.sh          (from repo root)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}/../.."
COA_COMPILER="${ROOT}/build/bin/coa-compiler"
COA_OPT="${ROOT}/build/bin/coa-opt"
INPUT="${SCRIPT_DIR}/model/resnet18.mlir"
OUTPUT_DIR="${SCRIPT_DIR}/output"
OUTPUT="${OUTPUT_DIR}/resnet18.vliw"

# ---- Sanity checks ----
if [[ ! -x "${COA_COMPILER}" ]]; then
    echo "[compile] ERROR: coa-compiler not found at ${COA_COMPILER}"
    echo "        Run  ./build.sh  from the repo root first (see top-level README)."
    exit 1
fi

if [[ ! -f "${INPUT}" ]]; then
    echo "[compile] ERROR: ${INPUT} not found."
    echo "        Run python import_mlir.py first."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[compile] Compiling ${INPUT} ..."
echo ""

"${COA_COMPILER}" \
    --output "${OUTPUT}" \
    --weight-base 0x08000000 \
    --bias-base   0xC0000000 \
    --act-base    0x10000000 \
    "${INPUT}"

echo ""
echo "[compile] Done -> ${OUTPUT}"
echo "[compile] Next: run  python verify.py"

# ---- Optional: emit lowered MLIR for inspection ----
if [[ -x "${COA_OPT}" ]]; then
    LOWERED="${OUTPUT_DIR}/resnet18_lowered.mlir"
    echo ""
    echo "[compile] Emitting lowered MLIR for inspection ..."
    "${COA_OPT}" \
        --coa-shape-infer \
        --coa-op-fusion \
        --coa-tiling \
        --coa-addr-assign \
        --coa-legalize \
        "${INPUT}" -o "${LOWERED}"
    echo "[compile] Lowered MLIR -> ${LOWERED}"
fi
