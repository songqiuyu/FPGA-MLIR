"""
提取 Bias
"""

import os
import re
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_weight_key(weights_dict, name):
    """查找 npz 中的键名"""
    if name in weights_dict.files:
        return name
    name_suffix = name.split('_')[-1]
    for k in weights_dict.files:
        if k.endswith(name_suffix):
            k_prefix = k[:-len(name_suffix)-1]
            name_prefix = '_'.join(name.split('_')[:-1])
            k_prefix_und = k_prefix.replace('.', '_')
            if k_prefix_und == name_prefix:
                return k
    return None


def parse_mlir_order(mlir_path):
    """解析 MLIR 获取算子顺序"""
    with open(mlir_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'%(\w+)\s*=\s*"coa\.qlinearconv"\(%(\w+),\s*%(\w+),\s*%(\w+)\)'
    matches = re.findall(pattern, content)
    ops = []
    for out_var, in_var, weight_var, bias_var in matches:
        ops.append({
            'out': out_var, 'input': in_var,
            'weight': weight_var, 'bias': bias_var
        })

    pattern = r'%(\w+)\s*=\s*"coa\.qgemm"\(%(\w+),\s*%(\w+),\s*%(\w+)\)'
    matches = re.findall(pattern, content)
    for out_var, in_var, weight_var, bias_var in matches:
        ops.append({
            'out': out_var, 'input': in_var,
            'weight': weight_var, 'bias': bias_var,
            'type': 'qgemm'
        })

    return ops


def extract():
    """提取 bias"""
    mlir_path = os.path.join(PROJECT_ROOT, "models", "mlir_models", "resnet18_quant_int8.mlir")
    npz_path = os.path.join(PROJECT_ROOT, "models", "intermediate", "resnet18_quant_int8.npz")
    output_dir = os.path.join(PROJECT_ROOT, "parameters")
    os.makedirs(output_dir, exist_ok=True)

    bin_path = os.path.join(output_dir, "bias.image")
    coe_path = os.path.join(output_dir, "bias.coe")
    txt_path = os.path.join(output_dir, "bias_offset.txt")

    ops = parse_mlir_order(mlir_path)
    weights = np.load(npz_path)
    bias_addr = 0

    with open(bin_path, 'wb') as f_bin, \
         open(coe_path, 'w', encoding='utf-8') as f_coe, \
         open(txt_path, 'w', encoding='utf-8') as f_txt:

        f_coe.write("// Bias values (int32 hex)\n")
        f_txt.write("# Bias 偏移地址映射\n")
        f_txt.write("-" * 70 + "\n")

        for i, op in enumerate(ops):
            b_key = find_weight_key(weights, op['bias'])
            if b_key is None:
                continue

            b = weights[b_key].astype(np.int32)
            m = b.shape[0]
            m16 = ((m + 15) >> 4) * 16

            # pad to 16
            b_padded = np.zeros(m16, dtype=np.int32)
            b_padded[:m] = b
            b_padded.tofile(f_bin)

            # coe
            for row in range((m16 + 15) // 16):
                start = row * 16
                end = min(start + 16, m16)
                row_hex = ''.join(f'{val & 0xffffffff:08x}' for val in b_padded[start:end])
                f_coe.write(row_hex + ",\n")

            f_txt.write(f"0x{bias_addr:08x}: {op['out']} | bias:{op['bias']} shape:({m}) pad:{m16}\n")
            bias_addr += b_padded.nbytes

        f_txt.write("-" * 70 + "\n")
        f_txt.write(f"# Total: {bias_addr} bytes\n")

    print(f"[Bias] {bin_path}, {bias_addr} bytes")


if __name__ == "__main__":
    extract()