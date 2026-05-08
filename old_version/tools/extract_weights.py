"""
提取权重 (weight only)
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

    # qlinearconv
    pattern = r'%(\w+)\s*=\s*"coa\.qlinearconv"\(%(\w+),\s*%(\w+),\s*%(\w+)\)'
    matches = re.findall(pattern, content)
    ops = []
    for out_var, in_var, weight_var, bias_var in matches:
        ops.append({
            'out': out_var, 'input': in_var,
            'weight': weight_var, 'bias': bias_var
        })

    # qgemm
    pattern = r'%(\w+)\s*=\s*"coa\.qgemm"\(%(\w+),\s*%(\w+),\s*%(\w+)\)'
    matches = re.findall(pattern, content)
    for out_var, in_var, weight_var, bias_var in matches:
        ops.append({
            'out': out_var, 'input': in_var,
            'weight': weight_var, 'bias': bias_var,
            'type': 'qgemm'
        })

    return ops


def extract_conv_weight(weights_dict, weight_name, bias_name, zero_point=0):
    """提取并转置权重 (OC,IC,H,W)->(OC,H,W,IC), pad IC to 32"""
    w_key = find_weight_key(weights_dict, weight_name)
    b_key = find_weight_key(weights_dict, bias_name)
    if w_key is None or b_key is None:
        return None, None, 0

    w = weights_dict[w_key]
    b = weights_dict[b_key]
    w_t = np.transpose(w, (0, 2, 3, 1))

    oc, h, w_sz, ic = w_t.shape
    ic_padded = ((ic + 31) // 32) * 32
    if ic_padded > ic:
        pad_width = ((0, 0), (0, 0), (0, 0), (0, ic_padded - ic))
        w_t = np.pad(w_t, pad_width, mode='constant', constant_values=zero_point)

    return w_t.astype(np.int8), b.astype(np.int32), ic_padded


def extract_gemm_weight(weights_dict, weight_name, bias_name):
    """提取 FC 权重, pad to 32"""
    w_key = find_weight_key(weights_dict, weight_name)
    b_key = find_weight_key(weights_dict, bias_name)
    if w_key is None or b_key is None:
        return None, None, 0

    w = weights_dict[w_key]
    b = weights_dict[b_key]
    w_t = np.transpose(w, (1, 0))

    in_dim = w_t.shape[0]
    in_padded = ((in_dim + 31) // 32) * 32
    if in_padded > in_dim:
        pad_width = ((0, in_padded - in_dim), (0, 0))
        w_t = np.pad(w_t, pad_width, mode='constant', constant_values=0)

    return w_t.astype(np.int8), b.astype(np.int32), in_padded


def extract():
    """提取权重"""
    mlir_path = os.path.join(PROJECT_ROOT, "models", "mlir_models", "resnet18_quant_int8.mlir")
    npz_path = os.path.join(PROJECT_ROOT, "models", "intermediate", "resnet18_quant_int8.npz")
    output_dir = os.path.join(PROJECT_ROOT, "parameters")
    os.makedirs(output_dir, exist_ok=True)

    bin_path = os.path.join(output_dir, "weight.image")
    txt_path = os.path.join(output_dir, "weight_offset.txt")

    ops = parse_mlir_order(mlir_path)
    weights = np.load(npz_path)
    weight_addr = 0

    with open(bin_path, 'wb') as f_bin, open(txt_path, 'w', encoding='utf-8') as f_txt:
        f_txt.write("# Weight 偏移地址映射\n")
        f_txt.write("-" * 70 + "\n")

        for i, op in enumerate(ops):
            op_type = op.get('type', 'qlinearconv')
            if op_type == 'qgemm':
                w, b, ic_padded = extract_gemm_weight(weights, op['weight'], op['bias'])
            else:
                w, b, ic_padded = extract_conv_weight(weights, op['weight'], op['bias'])

            if w is None:
                continue

            w.tofile(f_bin)
            w_size = w.nbytes

            f_txt.write(f"0x{weight_addr:08x}: {op_type} {op['input']} -> {op['out']} | "
                       f"weight:{op['weight']} shape:{w.shape} | ic_pad:{ic_padded}\n")

            weight_addr += w_size

        f_txt.write("-" * 70 + "\n")
        f_txt.write(f"# Total: {weight_addr} bytes\n")

    print(f"[Weight] {bin_path}, {weight_addr} bytes")


if __name__ == "__main__":
    extract()