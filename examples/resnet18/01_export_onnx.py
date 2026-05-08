"""
Step 1: ResNet-18 PTQ 量化 + ONNX 导出
---------------------------------------
若 legacy/models/onnx_models/resnet18_quant_int8.onnx 已存在则直接跳过。
否则：
  1. 加载 legacy/models/onnx_models/resnet18.onnx
  2. 用 legacy/datasets/calibration_data/ 中的标定数据做 post-training 量化
  3. 导出量化后的 INT8 ONNX 模型

依赖：onnx, onnxruntime（或 onnxruntime-extensions）
"""

import os
import sys
import numpy as np

ROOT     = os.path.join(os.path.dirname(__file__), '..', '..')
LEGACY   = os.path.join(ROOT, 'legacy')
ONNX_DIR = os.path.join(LEGACY, 'models', 'onnx_models')
CALIB_DIR= os.path.join(LEGACY, 'datasets', 'calibration_data')
OUT_ONNX = os.path.join(ONNX_DIR, 'resnet18_quant_int8.onnx')

sys.path.insert(0, os.path.join(LEGACY, 'coa_mlir'))
sys.path.insert(0, os.path.join(LEGACY, 'tools'))


def load_calib_data():
    """Load calibration numpy batches from legacy/datasets/."""
    batches = []
    for fname in sorted(os.listdir(CALIB_DIR)):
        if fname.endswith('.npy'):
            batches.append(np.load(os.path.join(CALIB_DIR, fname)))
    if not batches:
        raise FileNotFoundError(f"No .npy files found in {CALIB_DIR}")
    return np.concatenate(batches, axis=0)


def quantize_and_export():
    """Quantize ResNet-18 to INT8 and save as ONNX."""
    # ---- Check if already exists ----
    if os.path.exists(OUT_ONNX):
        print(f"[Step1] {OUT_ONNX} already exists, skipping quantization.")
        return OUT_ONNX

    float_onnx = os.path.join(ONNX_DIR, 'resnet18.onnx')
    if not os.path.exists(float_onnx):
        raise FileNotFoundError(
            f"Float ONNX not found: {float_onnx}\n"
            "Please place resnet18.onnx in legacy/models/onnx_models/")

    try:
        from quantization.calibrator import Calibrator
        calib_data = load_calib_data()
        print(f"[Step1] Loaded {len(calib_data)} calibration samples")

        calibrator = Calibrator(float_onnx)
        calibrator.collect_stats(calib_data)
        calibrator.export_quantized(OUT_ONNX)
        print(f"[Step1] Quantized ONNX saved to {OUT_ONNX}")

    except ImportError:
        # Fallback: use onnxruntime quantization
        print("[Step1] Calibrator not found, trying onnxruntime static quantization...")
        try:
            from onnxruntime.quantization import (
                quantize_static, CalibrationDataReader, QuantType)

            class _CalibReader(CalibrationDataReader):
                def __init__(self, data):
                    self._data = [{"input": d[None]} for d in data[:64]]
                    self._idx = 0
                def get_next(self):
                    if self._idx >= len(self._data):
                        return None
                    out = self._data[self._idx]
                    self._idx += 1
                    return out

            calib_data = load_calib_data()
            quantize_static(
                float_onnx, OUT_ONNX,
                calibration_data_reader=_CalibReader(calib_data),
                quant_format=None,  # QOperator format
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
            )
            print(f"[Step1] onnxruntime INT8 ONNX saved to {OUT_ONNX}")
        except ImportError:
            print("[Step1] ERROR: Neither Calibrator nor onnxruntime found.")
            print("  pip install onnxruntime")
            sys.exit(1)

    return OUT_ONNX


if __name__ == "__main__":
    out = quantize_and_export()
    print(f"\n[Step1] Done -> {out}")
    print("[Step1] Next: run  02_import_mlir.py")
