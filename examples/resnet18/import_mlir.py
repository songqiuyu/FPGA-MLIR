"""
import_mlir.py — 量化 ONNX → Level-1 COA MLIR 转换
--------------------------------------------------
调用 coa.onnx_importer 将 data/resnet18_quant_int8.onnx
转换为 Level-1 COA MLIR，写入 model/resnet18.mlir。

输出 MLIR 仅包含 Level-1 属性（ONNX 量化参数 + 算子图），
形状 / 分块 / 地址等将由 coa-compiler C++ 流水线填入。
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'model')
os.makedirs(OUT_DIR, exist_ok=True)

from coa.onnx_importer import onnx_to_coa_mlir

IN_ONNX  = os.path.join(DATA_DIR, 'resnet18_quant_int8.onnx')
OUT_MLIR = os.path.join(OUT_DIR, 'resnet18.mlir')


def import_to_mlir():
    if not os.path.exists(IN_ONNX):
        print(f"[import_mlir] ERROR: {IN_ONNX} not found.")
        print(f"  Run export_onnx.py first (output goes to {DATA_DIR})")
        sys.exit(1)

    print(f"[import_mlir] Converting {IN_ONNX} -> COA MLIR ...")
    try:
        mlir_text = onnx_to_coa_mlir(IN_ONNX, func_name="resnet18")
    except Exception as e:
        print(f"[import_mlir] ERROR: {e}")
        sys.exit(1)

    with open(OUT_MLIR, 'w', encoding='utf-8') as f:
        f.write(mlir_text)
    print(f"[import_mlir] COA MLIR written to {OUT_MLIR}")
    print(f"[import_mlir] Lines: {len(mlir_text.splitlines())}")
    return OUT_MLIR


if __name__ == "__main__":
    out = import_to_mlir()
    print(f"\n[import_mlir] Done -> {out}")
    print("[import_mlir] Next: run  bash compile.sh")
