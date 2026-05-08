"""
export_onnx.py — ResNet-18 PTQ 量化 + ONNX 导出
-------------------------------------------------
若 data/resnet18_quant_int8.onnx 已存在则直接跳过。
否则：
  1. 加载 data/resnet18.onnx（需手动放入）
  2. 用 data/calibration/*.npy 做 post-training 量化（可选）
  3. 导出量化后的 INT8 ONNX 模型到 data/resnet18_quant_int8.onnx

依赖：onnx, onnxruntime
"""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

CALIB_DIR = os.path.join(DATA_DIR, 'calibration')
IN_ONNX   = os.path.join(DATA_DIR, 'resnet18.onnx')
OUT_ONNX  = os.path.join(DATA_DIR, 'resnet18_quant_int8.onnx')


def load_calib_data():
    """Load calibration numpy batches from data/calibration/."""
    batches = []
    if os.path.isdir(CALIB_DIR):
        for fname in sorted(os.listdir(CALIB_DIR)):
            if fname.endswith('.npy'):
                batches.append(np.load(os.path.join(CALIB_DIR, fname)))
    if not batches:
        raise FileNotFoundError(
            f"No .npy files found in {CALIB_DIR}\n"
            "  Place calibration data as data/calibration/*.npy")
    return np.concatenate(batches, axis=0)


def quantize_and_export():
    """Quantize ResNet-18 to INT8 and save as ONNX."""
    if os.path.exists(OUT_ONNX):
        print(f"[export_onnx] {OUT_ONNX} already exists, skipping quantization.")
        return OUT_ONNX

    float_onnx = IN_ONNX
    if not os.path.exists(float_onnx):
        raise FileNotFoundError(
            f"Float ONNX not found: {float_onnx}\n"
            "  Place resnet18.onnx in examples/resnet18/data/")

    try:
        calib_data = load_calib_data()
        print(f"[export_onnx] Loaded {len(calib_data)} calibration samples")
        _quantize_with_onnxruntime(float_onnx, calib_data)
    except FileNotFoundError:
        print("[export_onnx] No calibration data; trying static quantization with random data...")
        calib_data = np.random.randint(-128, 128, (64, 3, 224, 224), dtype=np.int8).astype(np.float32) / 128
        _quantize_with_onnxruntime(float_onnx, calib_data)
    return OUT_ONNX


def _quantize_with_onnxruntime(float_onnx: str, calib_data):
    """Run onnxruntime static INT8 quantization (QOperator format)."""
    try:
        from onnxruntime.quantization import (
            quantize_static, CalibrationDataReader, QuantType, QuantFormat)
        from onnxruntime.quantization.preprocess import quant_pre_process
    except ImportError:
        print("[export_onnx] ERROR: onnxruntime not found.")
        print("  pip install onnxruntime")
        sys.exit(1)

    # Pre-process: fold BN, add missing biases, run shape inference
    preprocessed = float_onnx.replace(".onnx", "_prep.onnx")
    print("[export_onnx] Pre-processing model (shape inference + BN fold) ...")
    quant_pre_process(float_onnx, preprocessed, skip_optimization=False)

    class _CalibReader(CalibrationDataReader):
        def __init__(self, data):
            self._data = [{"input": d[None].astype(np.float32)} for d in data[:64]]
            self._idx = 0
        def get_next(self):
            if self._idx >= len(self._data):
                return None
            out = self._data[self._idx]
            self._idx += 1
            return out

    print("[export_onnx] Running onnxruntime static INT8 quantization ...")
    quantize_static(
        preprocessed, OUT_ONNX,
        calibration_data_reader=_CalibReader(calib_data),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    # Remove temp file
    if os.path.exists(preprocessed):
        os.remove(preprocessed)
    print(f"[export_onnx] INT8 ONNX (QOperator) saved to {OUT_ONNX}")


if __name__ == "__main__":
    out = quantize_and_export()
    print(f"\n[export_onnx] Done -> {out}")
    print("[export_onnx] Next: run  python import_mlir.py")
