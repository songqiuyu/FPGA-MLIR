"""
生成测试输入 (input data)
支持: 量化 + 转置 + pad to 32
"""

import os
import re
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_input_params(mlir_path):
    """解析 MLIR 获取输入参数"""
    with open(mlir_path, 'r', encoding='utf-8') as f:
        content = f.read()

    scale_match = re.search(r'quantizelinear.*?scale\s*=\s*([\d.]+)', content)
    zp_match = re.search(r'quantizelinear.*?zp\s*=\s*(-?[\d.]+)', content)

    if scale_match:
        return {
            'scale': float(scale_match.group(1)),
            'zero_point': int(zp_match.group(1)) if zp_match else 0,
        }
    return None


def extract():
    """提取输入"""
    mlir_path = os.path.join(PROJECT_ROOT, "models", "mlir_models", "resnet18_quant_int8.mlir")
    output_dir = os.path.join(PROJECT_ROOT, "parameters")
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入参数 - 从MLIR读取
    params = parse_input_params(mlir_path)
    scale = params['scale']
    zp = params['zero_point']
    print(f"Input params: scale={scale}, zp={zp}")

    # 输入形状
    input_shape = [1, 3, 224, 224]
    print(f"Input shape: {input_shape}")

    # 读取浮点测试数据
    input_bin = os.path.join(PROJECT_ROOT, "images", "input_float.bin")
    input_data = np.fromfile(input_bin, dtype=np.float32).reshape(input_shape)
    print(f"Loaded: {input_data.shape}, dtype={input_data.dtype}")

    # 1. 量化: round(data / scale) + zero_point
    input_quantized = np.round(input_data / scale) + zp
    input_quantized = np.clip(input_quantized, -128, 127).astype(np.int8)
    print(f"Quantized: shape={input_quantized.shape}, dtype={input_quantized.dtype}")

    # 2. 转置: (N, C, H, W) -> (N, H, W, C)
    # (1, 3, 224, 224) -> (1, 224, 224, 3)
    input_t = np.transpose(input_quantized, (0, 2, 3, 1))
    print(f"Transposed: {input_t.shape}")

    # 3. Pad C: 3 -> 32 (用 zero_point 填充)
    n, h, w, c = input_t.shape
    c_padded = ((c + 31) // 32) * 32
    if c_padded > c:
        pad_width = ((0, 0), (0, 0), (0, 0), (0, c_padded - c))
        input_t = np.pad(input_t, pad_width, mode='constant', constant_values=zp)
    print(f"Padded: {input_t.shape}")

    # 4. 写入二进制 (按 32 列分块)
    # 实际存储: 每行 32 字节 (N, H, W, 32) 依次存储
    bin_path = os.path.join(output_dir, "input.image")
    with open(bin_path, 'wb') as f:
        # 遍历 (N, H, W, 32)
        for n_idx in range(input_t.shape[0]):
            for h_idx in range(input_t.shape[1]):
                for w_idx in range(input_t.shape[2]):
                    row_data = input_t[n_idx, h_idx, w_idx, :]
                    row_data.tofile(f)

    # 记录
    txt_path = os.path.join(output_dir, "input_offset.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("# Input 参数\n")
        f.write(f"# scale: {scale}\n")
        f.write(f"# zero_point: {zp}\n")
        f.write(f"# original shape: {input_shape}\n")
        f.write(f"# after transpose + pad: {input_t.shape}\n")
        f.write(f"0x00000000: input | shape: {input_t.shape}, dtype: int8\n")

    print(f"[Input] {bin_path}, {os.path.getsize(bin_path)} bytes")


if __name__ == "__main__":
    extract()