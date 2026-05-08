"""
make_float_onnx.py — 用 torchvision 导出 ResNet-18 浮点 ONNX 到 data/resnet18.onnx
需要 torch + torchvision + onnx（已在 ggml 环境中安装）。
"""

import os
import sys

import torch
import torchvision.models as models
import onnx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

OUT_ONNX = os.path.join(DATA_DIR, 'resnet18.onnx')

if os.path.exists(OUT_ONNX):
    print(f"[make_float_onnx] Already exists: {OUT_ONNX}")
    sys.exit(0)

print("[make_float_onnx] Loading torchvision ResNet-18 (pretrained=False) ...")
model = models.resnet18(weights=None)
model.eval()

dummy = torch.zeros(1, 3, 224, 224)

print(f"[make_float_onnx] Exporting to {OUT_ONNX} ...")
torch.onnx.export(
    model, dummy, OUT_ONNX,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=13,
)

# Verify
onnx.checker.check_model(OUT_ONNX)
print(f"[make_float_onnx] OK  size={os.path.getsize(OUT_ONNX)//1024} KB")
print("[make_float_onnx] Next: python export_onnx.py")
