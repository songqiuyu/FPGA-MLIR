"""
计算weight和bias的地址
"""

import numpy as np
import re

npz_path = "d:/PROJECT/FPGA-MLIR/models/intermediate/resnet18_quant_int8.npz"
mlir_path = "d:/PROJECT/FPGA-MLIR/models/mlir_models/resnet18_quant_int8.mlir"

npz = np.load(npz_path, allow_pickle=True)

# 读取MLIR获取算子顺序
with open(mlir_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 提取所有qlinearconv
pattern = r'%(\w+)\s*=\s*"coa\.qlinearconv"\(%[^,]+,\s*%(\w+),\s*%(\w+)\)'
matches = re.findall(pattern, content)

WEIGHT_BASE = 0x8000000
BIAS_BASE = 0xC0000000

print("=" * 80)
print("Weight地址分配 (基地址: 0x8000000)")
print("=" * 80)
print(f"{'层':<8} {'Weight Name':<12} {'Shape':<20} {'Size':<10} {'Addr':<12} {'Bias':<8} {'Bias Addr'}")
print("-" * 80)

weight_addr = WEIGHT_BASE
bias_addr = BIAS_BASE

for out_var, weight_var, bias_var in matches:
    if weight_var in npz.files:
        w = npz[weight_var]
        b = npz[bias_var]

        # 计算weight size: OC * H * W * padded_IC
        oc, ic, h, w_sz = w.shape
        ic_padded = ((ic + 31) // 32) * 32
        w_size = oc * h * w_sz * ic_padded
        b_size = len(b) * 4  # bias通常是int32, 4 bytes

        print(f"{out_var:<8} {weight_var:<12} {str(w.shape):<20} {w_size:<10} 0x{weight_addr:x} {b_size:<8} 0x{bias_addr:x}")

        weight_addr += w_size
        bias_addr += b_size

# FC层
if 'fc.weight_quantized' in npz.files:
    w = npz['fc.weight_quantized']
    b = npz['fc.bias_quantized']
    # FC: (out, in) -> 转置后 (in_padded, out)
    in_dim, out_dim = w.shape
    in_padded = ((in_dim + 31) // 32) * 32
    w_size = in_padded * out_dim
    b_size = out_dim * 4

    print(f"{'fc':<8} {'fc.weight':<12} {str(w.shape):<20} {w_size:<10} 0x{weight_addr:x} {b_size:<8} 0x{bias_addr:x}")

print("-" * 80)
print(f"Total weight size: 0x{weight_addr - WEIGHT_BASE:x}")
print(f"Total bias size: 0x{bias_addr - BIAS_BASE:x}")