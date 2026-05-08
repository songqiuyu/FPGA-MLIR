"""
生成所有层的 ReLU 激活函数查找表
一个文件包含所有层 LUT (每层 256 bytes)
"""

import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_mlir_layers(mlir_path):
    """解析 MLIR 获取需要 ReLU 的层"""
    with open(mlir_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 找所有 qlinearconv 输出变量
    layers = []
    lines = content.split('\n')
    for line in lines:
        if 'qlinearconv' in line:
            # 提取输出变量名, 如 %192_quantized
            m = re.search(r'%(\w+)\s*=\s*"coa\.qlinearconv"', line)
            if m:
                out_var = m.group(1)

                # 提取 scale/zp
                in_scale = re.search(r'in_scale\s*=\s*([\d.]+)', line)
                in_zp = re.search(r'in_zp\s*=\s*(-?[\d.]+)', line)
                out_scale = re.search(r'out_scale\s*=\s*([\d.]+)', line)
                out_zp = re.search(r'out_zp\s*=\s*(-?[\d.]+)', line)

                layers.append({
                    'output': out_var,
                    'in_scale': float(in_scale.group(1)) if in_scale else None,
                    'in_zp': int(in_zp.group(1)) if in_zp else None,
                    'out_scale': float(out_scale.group(1)) if out_scale else None,
                    'out_zp': int(out_zp.group(1)) if out_zp else None,
                })

    return layers


def generate_relu_lut():
    output_dir = os.path.join(PROJECT_ROOT, "parameters")
    os.makedirs(output_dir, exist_ok=True)

    mlir_path = os.path.join(PROJECT_ROOT, "models", "mlir_models", "resnet18_quant_int8.mlir")
    layers = parse_mlir_layers(mlir_path)

    num_layers = len(layers)
    lut_size = 256
    total_size = num_layers * lut_size

    bin_path = os.path.join(output_dir, "relu.image")
    coe_path = os.path.join(output_dir, "relu.coe")
    txt_path = os.path.join(output_dir, "relu_offset.txt")

    # 写入 binary: 每层一个 f(x)=x 的 LUT
    with open(bin_path, 'wb') as f:
        for layer_idx in range(num_layers):
            for i in range(256):
                int8_val = i if i < 128 else i - 256
                f.write(int8_val.to_bytes(1, byteorder='little', signed=True))

    # 写入 coe: 每行 16 个索引
    with open(coe_path, 'w', encoding='utf-8') as f:
        f.write("// ReLU LUT (all layers)\n")
        f.write("// Format: 16 values per line\n")
        for row in range(256 // 16):
            start = row * 16
            end = start + 16
            row_hex = ''.join(f'{i:02x}' for i in range(start, end))
            f.write(row_hex + ",\n")

    # 写入偏移文件
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("# ReLU LUT 偏移地址映射\n")
        f.write("# Format: 地址: 层索引 输出变量 [in_scale,in_zp,out_scale,out_zp]\n")
        f.write("-" * 70 + "\n")

        offset = 0
        for i, layer in enumerate(layers):
            attrs = layer['in_scale'], layer['in_zp'], layer['out_scale'], layer['out_zp']
            line = f"0x{offset:08x}: layer_{i} {layer['output']} | in_s={attrs[0]:.6f} in_zp={attrs[1]} out_s={attrs[2]:.6f} out_zp={attrs[3]}\n"
            f.write(line)
            offset += lut_size

        f.write("-" * 70 + "\n")
        f.write(f"# Total: {total_size} bytes ({num_layers} layers x {lut_size} bytes)\n")

    print(f"[Done] ReLU image: {bin_path}")
    print(f"[Done] ReLU coe: {coe_path}")
    print(f"[Done] Offset file: {txt_path}")
    print(f"  Total: {total_size} bytes, {num_layers} layers")


if __name__ == "__main__":
    print("=" * 60)
    print("[ReLU LUT Generator]")
    print("=" * 60)
    generate_relu_lut()
    print("=" * 60)