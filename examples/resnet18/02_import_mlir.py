"""
Step 2: 量化 ONNX → COA MLIR 转换
------------------------------------
调用 legacy/coa_mlir/frontend/ 将 resnet18_quant_int8.onnx
转换为 Level-1 COA MLIR，写入 model/resnet18.mlir。

输出 MLIR 仅包含 Level-1 属性（ONNX 量化参数 + 算子图），
形状 / 分块 / 地址等将由 coa-compiler C++ 流水线填入。
"""

import os
import sys

ROOT     = os.path.join(os.path.dirname(__file__), '..', '..')
LEGACY   = os.path.join(ROOT, 'legacy')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(LEGACY, 'coa_mlir'))
sys.path.insert(0, os.path.join(LEGACY, 'tools'))

IN_ONNX  = os.path.join(LEGACY, 'models', 'onnx_models', 'resnet18_quant_int8.onnx')
OUT_MLIR = os.path.join(OUT_DIR, 'resnet18.mlir')


def import_to_mlir():
    if not os.path.exists(IN_ONNX):
        print(f"[Step2] ERROR: {IN_ONNX} not found.")
        print("  Run 01_export_onnx.py first, or copy from legacy/models/")
        sys.exit(1)

    try:
        from frontend.importer import ONNXImporter
        from frontend.mlir_gen  import MLIRGen

        print(f"[Step2] Importing {IN_ONNX} ...")
        importer = ONNXImporter(IN_ONNX)
        graph    = importer.import_graph()
        print(f"[Step2] Found {len(graph.nodes)} ops")

        mlir_gen = MLIRGen(graph)
        mlir_text = mlir_gen.generate()

        with open(OUT_MLIR, 'w', encoding='utf-8') as f:
            f.write(mlir_text)
        print(f"[Step2] COA MLIR written to {OUT_MLIR}")
        print(f"[Step2] Lines: {len(mlir_text.splitlines())}")

    except ImportError as e:
        print(f"[Step2] Import error: {e}")
        print("  Falling back to legacy onnx_to_mlir.py ...")
        _fallback_onnx_to_mlir()

    return OUT_MLIR


def _fallback_onnx_to_mlir():
    """Call legacy tools/onnx_to_mlir.py as a subprocess fallback."""
    import subprocess
    script = os.path.join(LEGACY, 'tools', 'onnx_to_mlir.py')
    if not os.path.exists(script):
        print(f"[Step2] ERROR: {script} not found.")
        sys.exit(1)

    python = sys.executable
    ret = subprocess.run(
        [python, script, IN_ONNX, '--output', OUT_MLIR],
        capture_output=False
    )
    if ret.returncode != 0:
        print(f"[Step2] onnx_to_mlir.py failed (exit {ret.returncode})")
        sys.exit(1)
    print(f"[Step2] Fallback OK -> {OUT_MLIR}")


if __name__ == "__main__":
    out = import_to_mlir()
    print(f"\n[Step2] Done -> {out}")
    print("[Step2] Next: run  03_compile.bat")
