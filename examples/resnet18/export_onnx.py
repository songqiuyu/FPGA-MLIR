"""
export_onnx.py — ResNet-18 PTQ 量化 + ONNX 导出
-------------------------------------------------
若 data/resnet18_quant_int8.onnx 已存在则直接跳过。
否则：
  1. 加载 data/resnet18.onnx（需手动放入）
  2. 用 data/calibration/*.npy 做 post-training 量化（可选）
  3. 导出量化后的 INT8 ONNX 模型到 data/resnet18_quant_int8.onnx

使用 coa.quantize 自研量化工具（不依赖 onnxruntime）。
依赖：onnx, numpy
"""

import os
import sys
import numpy as np

# Ensure project root is on sys.path so `coa` package can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')
DATA_DIR  = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

IN_ONNX  = os.path.join(MODEL_DIR, 'resnet18.onnx')
OUT_ONNX = os.path.join(DATA_DIR,  'resnet18_quant_int8.onnx')

# Calibration data search order
_CALIB_CANDIDATES = [
    os.path.join(DATA_DIR, 'calibration'),
    os.path.join(PROJECT_ROOT, 'legacy', 'datasets', 'calibration_data'),
]


def load_calib_data():
    """Load calibration numpy batches from data/calibration/ or legacy path."""
    for calib_dir in _CALIB_CANDIDATES:
        if not os.path.isdir(calib_dir):
            continue
        batches = [np.load(os.path.join(calib_dir, f))
                   for f in sorted(os.listdir(calib_dir))
                   if f.endswith('.npy')]
        if batches:
            print(f"[export_onnx] Calibration data: {calib_dir}")
            return np.concatenate(batches, axis=0)
    raise FileNotFoundError(
        "No .npy calibration files found. Searched:\n"
        + "\n".join(f"  {d}" for d in _CALIB_CANDIDATES))


def quantize_and_export(
    act_format: str = "int8",
    calibration: str = "minmax",
    smooth_alpha: float = None,
    use_gptq: bool = False,
):
    """Quantize ResNet-18 to INT8 and save as ONNX.

    Parameters
    ----------
    act_format : "int8" (symmetric, default) or "uint8" (asymmetric).
    calibration : "minmax" | "percentile" | "entropy".
    smooth_alpha : SmoothQuant α (0-1), None to disable.
    use_gptq : enable GPTQ-style weight error correction.
    """
    if not os.path.exists(IN_ONNX):
        raise FileNotFoundError(
            f"Float ONNX not found: {IN_ONNX}\n"
            "  Place resnet18.onnx in examples/resnet18/model/")

    try:
        calib_data = load_calib_data()
        print(f"[export_onnx] Loaded {len(calib_data)} calibration samples")
    except FileNotFoundError:
        print("[export_onnx] No calibration data; using 64 random samples ...")
        calib_data = np.random.randn(64, 3, 224, 224).astype(np.float32)

    from coa.quantize import quantize_onnx

    quantize_onnx(
        IN_ONNX,
        OUT_ONNX,
        calib_data,
        act_format=act_format,
        weight_per_channel=False,
        calibration=calibration,
        smooth_alpha=smooth_alpha,
        use_gptq=use_gptq,
    )
    return OUT_ONNX


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ResNet-18 PTQ quantization")
    parser.add_argument("--act-format", choices=["int8", "uint8"], default="int8",
                        help="Activation format: int8 (default) or uint8")
    parser.add_argument("--calibration", choices=["minmax", "percentile", "entropy"],
                        default="minmax", help="Calibration method")
    parser.add_argument("--smooth-alpha", type=float, default=None,
                        help="SmoothQuant alpha (0-1)")
    parser.add_argument("--gptq", action="store_true", default=False,
                        help="Enable GPTQ weight correction")
    args = parser.parse_args()

    out = quantize_and_export(
        act_format=args.act_format,
        calibration=args.calibration,
        smooth_alpha=args.smooth_alpha,
        use_gptq=args.gptq,
    )
    print(f"\n[export_onnx] Done -> {out}")
    print("[export_onnx] Next: run  python -m examples.resnet18.import_mlir")
