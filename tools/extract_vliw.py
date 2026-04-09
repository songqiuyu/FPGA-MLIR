"""
生成 VLIW 指令
根据 MLIR 信息
operator: 0=卷积, 1=池化, 2=上采样, 3=add
"""

import os
import re
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 算子类型映射
OperatorMap = {
    'qlinearconv': 0,    # 卷积
    'qgemm': 0,          # GEMM
    'maxpool': 1,         # 池化
    'qlinearadd': 3,      # Add (残差)
    'qlinearglobalaveragepool': 2,  # 上采样
}


def parse_all_layers(mlir_path):
    """解析所有操作层（卷积、GEMM、池化、Add）"""
    with open(mlir_path, 'r', encoding='utf-8') as f:
        content = f.read()

    layers = []

    # 1. qlinearconv
    pattern = r'%(\w+)\s*=\s*"coa\.qlinearconv"\(%(\w+),\s*%(\w+),\s*%(\w+)\)\s*\{([^}]+)\}'
    for out_var, in_var, weight_var, bias_var, attrs in re.findall(pattern, content):
        in_scale = re.search(r'in_scale\s*=\s*([\d.]+)', attrs)
        in_zp = re.search(r'in_zp\s*=\s*(-?[\d.]+)', attrs)
        w_scale = re.search(r'weight_scale\s*=\s*([\d.]+)', attrs)
        w_zp = re.search(r'weight_zp\s*=\s*(-?[\d.]+)', attrs)
        out_scale = re.search(r'out_scale\s*=\s*([\d.]+)', attrs)
        out_zp = re.search(r'out_zp\s*=\s*(-?[\d.]+)', attrs)
        dilations = re.search(r'dilations\s*=\s*\[(\d+),\s*(\d+)\]', attrs)
        group = re.search(r'group\s*=\s*(\d+)', attrs)
        kernel = re.search(r'kernel_shape\s*=\s*\[(\d+),\s*(\d+)\]', attrs)
        pads = re.search(r'pads\s*=\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)', attrs)
        strides = re.search(r'strides\s*=\s*\[(\d+),\s*(\d+)\]', attrs)

        layers.append({
            'output': out_var,
            'input': in_var,
            'weight': weight_var,
            'bias': bias_var,
            'in_scale': float(in_scale.group(1)) if in_scale else 1.0,
            'in_zp': int(in_zp.group(1)) if in_zp else 0,
            'w_scale': float(w_scale.group(1)) if w_scale else 1.0,
            'w_zp': int(w_zp.group(1)) if w_zp else 0,
            'out_scale': float(out_scale.group(1)) if out_scale else 1.0,
            'out_zp': int(out_zp.group(1)) if out_zp else 0,
            'dilations': [int(dilations.group(1)), int(dilations.group(2))] if dilations else [1, 1],
            'group': int(group.group(1)) if group else 1,
            'kernel': [int(kernel.group(1)), int(kernel.group(2))] if kernel else [3, 3],
            'pads': [int(pads.group(i)) for i in range(1, 5)] if pads else [0, 0, 0, 0],
            'strides': [int(strides.group(1)), int(strides.group(2))] if strides else [1, 1],
            'type': 'qlinearconv'
        })

    # 2. qgemm
    pattern = r'%(\w+)\s*=\s*"coa\.qgemm"\(%(\w+),\s*%(\w+),\s*%(\w+)\)\s*\{([^}]+)\}'
    for out_var, in_var, weight_var, bias_var, attrs in re.findall(pattern, content):
        a_scale = re.search(r'a_scale\s*=\s*([\d.]+)', attrs)
        a_zp = re.search(r'a_zp\s*=\s*(-?[\d.]+)', attrs)
        b_scale = re.search(r'b_scale\s*=\s*([\d.]+)', attrs)
        b_zp = re.search(r'b_zp\s*=\s*(-?[\d.]+)', attrs)
        out_scale = re.search(r'out_scale\s*=\s*([\d.]+)', attrs)
        out_zp = re.search(r'out_zp\s*=\s*(-?[\d.]+)', attrs)
        transB = re.search(r'transB\s*=\s*(\d+)', attrs)

        layers.append({
            'output': out_var,
            'input': in_var,
            'weight': weight_var,
            'bias': bias_var,
            'in_scale': float(a_scale.group(1)) if a_scale else 1.0,
            'in_zp': int(a_zp.group(1)) if a_zp else 0,
            'w_scale': float(b_scale.group(1)) if b_scale else 1.0,
            'w_zp': int(b_zp.group(1)) if b_zp else 0,
            'out_scale': float(out_scale.group(1)) if out_scale else 1.0,
            'out_zp': int(out_zp.group(1)) if out_zp else 0,
            'transB': int(transB.group(1)) if transB else 0,
            'type': 'qgemm'
        })

    return layers


def parse_mlir_conv_layers(mlir_path):
    """解析 MLIR 获取所有卷积层"""
    with open(mlir_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 找 qlinearconv
    pattern = r'%(\w+)\s*=\s*"coa\.qlinearconv"\(%(\w+),\s*%(\w+),\s*%(\w+)\)\s*\{([^}]+)\}'
    matches = re.findall(pattern, content)

    layers = []
    for out_var, in_var, weight_var, bias_var, attrs in matches:
        # 提取量化参数
        in_scale = re.search(r'in_scale\s*=\s*([\d.]+)', attrs)
        in_zp = re.search(r'in_zp\s*=\s*(-?[\d.]+)', attrs)
        w_scale = re.search(r'weight_scale\s*=\s*([\d.]+)', attrs)
        w_zp = re.search(r'weight_zp\s*=\s*(-?[\d.]+)', attrs)
        out_scale = re.search(r'out_scale\s*=\s*([\d.]+)', attrs)
        out_zp = re.search(r'out_zp\s*=\s*(-?[\d.]+)', attrs)

        # 提取卷积参数
        dilations = re.search(r'dilations\s*=\s*\[(\d+),\s*(\d+)\]', attrs)
        group = re.search(r'group\s*=\s*(\d+)', attrs)
        kernel = re.search(r'kernel_shape\s*=\s*\[(\d+),\s*(\d+)\]', attrs)
        pads = re.search(r'pads\s*=\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', attrs)
        strides = re.search(r'strides\s*=\s*\[(\d+),\s*(\d+)\]', attrs)

        layers.append({
            'output': out_var,
            'input': in_var,
            'weight': weight_var,
            'bias': bias_var,
            'in_scale': float(in_scale.group(1)) if in_scale else 1.0,
            'in_zp': int(in_zp.group(1)) if in_zp else 0,
            'w_scale': float(w_scale.group(1)) if w_scale else 1.0,
            'w_zp': int(w_zp.group(1)) if w_zp else 0,
            'out_scale': float(out_scale.group(1)) if out_scale else 1.0,
            'out_zp': int(out_zp.group(1)) if out_zp else 0,
            'dilations': [int(dilations.group(1)), int(dilations.group(2))] if dilations else [1, 1],
            'group': int(group.group(1)) if group else 1,
            'kernel': [int(kernel.group(1)), int(kernel.group(2))] if kernel else [3, 3],
            'pads': [int(pads.group(i)) for i in range(1, 5)] if pads else [0, 0, 0, 0],
            'strides': [int(strides.group(1)), int(strides.group(2))] if strides else [1, 1],
        })

    return layers


def parse_gemm_layers(mlir_path):
    """解析 MLIR 获取 GEMM/全连接层"""
    with open(mlir_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'%(\w+)\s*=\s*"coa\.qgemm"\(%(\w+),\s*%(\w+),\s*%(\w+)\)\s*\{([^}]+)\}'
    matches = re.findall(pattern, content)

    layers = []
    for out_var, in_var, weight_var, bias_var, attrs in matches:
        # 提取量化参数
        a_scale = re.search(r'a_scale\s*=\s*([\d.]+)', attrs)
        a_zp = re.search(r'a_zp\s*=\s*(-?[\d.]+)', attrs)
        b_scale = re.search(r'b_scale\s*=\s*([\d.]+)', attrs)
        b_zp = re.search(r'b_zp\s*=\s*(-?[\d.]+)', attrs)
        out_scale = re.search(r'out_scale\s*=\s*([\d.]+)', attrs)
        out_zp = re.search(r'out_zp\s*=\s*(-?[\d.]+)', attrs)
        transB = re.search(r'transB\s*=\s*(\d+)', attrs)

        layers.append({
            'output': out_var,
            'input': in_var,
            'weight': weight_var,
            'bias': bias_var,
            'in_scale': float(a_scale.group(1)) if a_scale else 1.0,
            'in_zp': int(a_zp.group(1)) if a_zp else 0,
            'w_scale': float(b_scale.group(1)) if b_scale else 1.0,
            'w_zp': int(b_zp.group(1)) if b_zp else 0,
            'out_scale': float(out_scale.group(1)) if out_scale else 1.0,
            'out_zp': int(out_zp.group(1)) if out_zp else 0,
            'transB': int(transB.group(1)) if transB else 0,
            'type': 'qgemm'
        })

    return layers


def load_address_mapping(weight_offset_file):
    """加载权重地址映射 - 从 offset 文件"""
    addresses = {}
    try:
        with open(weight_offset_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('-'):
                    continue
                # 格式: 0x00000000: qlinearconv input -> 192 | weight:193 ...
                m = re.match(r'0x([0-9a-fA-F]+):', line)
                if m:
                    addr = int(m.group(1), 16)
                    # 提取 weight 名称
                    w_match = re.search(r'weight:(\S+)', line)
                    if w_match:
                        addresses[w_match.group(1)] = addr
    except FileNotFoundError:
        # 如果文件不存在，使用默认地址计算
        pass
    return addresses


def load_bias_mapping(bias_offset_file):
    """加载 bias 地址映射"""
    addresses = {}
    try:
        with open(bias_offset_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('-'):
                    continue
                m = re.match(r'0x([0-9a-fA-F]+):', line)
                if m:
                    addr = int(m.group(1), 16)
                    if 'bias:' in line:
                        b_match = re.search(r'bias:(\S+)', line)
                        if b_match:
                            addresses[b_match.group(1)] = addr
    except FileNotFoundError:
        pass
    return addresses


def calc_quant_factor(in_scale, w_scale, out_scale):
    """计算量化因子: factor = in_scale * w_scale / out_scale"""
    factor = in_scale * w_scale / out_scale
    # 转换为定点数 (这里简化处理)
    return int(factor * (1 << 30))  # 34 bits


def generate_vliw_binary(layer_info, input_addr, weight_addr, bias_addr, output_addr, relu_addr, op_type='qlinearconv'):
    """生成 VLIW 二进制"""
    # 获取 operator 类型
    operator = OperatorMap.get(op_type, 0)

    # 基本字段
    fields = [
        # (value, bits)
        (operator, 8),      # operator
        (input_addr, 36),   # DDR_x1_address
        (weight_addr, 36),  # DDR_x2_address
        (bias_addr, 11),   # Bias_source_address
        (output_addr, 36),  # Compute_Result_dest_address
        (relu_addr, 8),    # Activate_LUT_address
        (224, 11),   # R (输出高)
        (224, 11),   # C (输出宽)
        (64, 12),    # M (输出通道)
        (3, 12),     # N (输入通道)
        (224, 11),   # R0 (输入高)
        (224, 11),   # C0 (输入宽)
        (64, 12),    # sM_concat
        (64, 12),    # M_concat
        (layer_info.get('in_zp', 0), 8),   # Quant_x1_z
        (layer_info.get('w_zp', 0), 8),     # Quant_x2_z
        (layer_info.get('out_zp', 0), 8),    # Quant_y_z
        (layer_info.get('pads', [0,0,0,0])[0], 3),   # Conv_pad
        (layer_info.get('kernel', [3,3])[0], 5),           # Conv_kernel
        (layer_info.get('strides', [1,1])[0], 3),    # Conv_stride
        (layer_info.get('dilations', [1,1])[0], 3),  # Conv_dilation
        (224, 11),   # Conv_tR
        (224, 11),   # Conv_tC
        (64, 12),    # Conv_tM
        (3, 12),     # Conv_tN
        (0, 2),      # Conv_permuteR
        (0, 2),      # Conv_permuteC
        (0, 2),      # Conv_permuteM
        (0, 2),      # Conv_permuteN
    ]

    # 添加量化因子
    qf = calc_quant_factor(
        layer_info.get('in_scale', 1.0),
        layer_info.get('w_scale', 1.0),
        layer_info.get('out_scale', 1.0)
    )
    fields.append((qf, 34))   # Conv_quant_factor
    fields.append((0, 32))   # Conv_quant_factor2
    fields.append((0, 127))  # high bits

    # 构建比特流
    bits = []
    for value, num_bits in fields:
        for i in range(num_bits):
            bits.append((value >> i) & 1)

    # 转换为字节
    result = 0
    bit_count = 0
    out_bytes = []

    for bit in bits:
        result |= (bit << bit_count)
        bit_count += 1
        if bit_count == 8:
            out_bytes.append(result & 0xFF)
            result = 0
            bit_count = 0

    while len(out_bytes) < 64:
        out_bytes.append(0)

    return bytes(out_bytes[:64])


def extract():
    """生成 VLIW 指令"""
    mlir_path = os.path.join(PROJECT_ROOT, "models", "mlir_models", "resnet18_quant_int8.mlir")
    output_dir = os.path.join(PROJECT_ROOT, "parameters")
    weight_offset_file = os.path.join(output_dir, "weight_offset.txt")
    bias_offset_file = os.path.join(output_dir, "bias_offset.txt")

    # 解析所有层信息
    all_layers = parse_all_layers(mlir_path)

    # 加载地址映射
    weight_addrs = load_address_mapping(weight_offset_file)
    bias_addrs = load_bias_mapping(bias_offset_file)

    print(f"Found {len(all_layers)} layers")

    # ====== Step 1: 建立依赖关系 ======
    # 统计每个输出被多少层使用
    output_usage = {}  # output_name -> 使用次数
    for layer in all_layers:
        inp = layer['input']
        if inp not in output_usage:
            output_usage[inp] = 0
        output_usage[inp] += 1

    print(f"Output usage: {output_usage}")

    # ====== Step 2: 地址池管理 ======
    # 地址池: 从 0x10000000 开始，每次 +0x1000000，最大 0x20000000
    addr_pool_start = 0x10000000
    addr_pool_end = 0x20000000
    addr_increment = 0x1000000

    # 已分配的输出地址 (变量名 -> 地址)
    output_addrs = {}
    # 可回收的地址列表
    free_addrs = []
    # 当前可用地址
    current_addr = addr_pool_start
    # 已分配的地址集合（用于去重）
    used_addrs = set()

    # ====== Step 3: 生成指令 ======
    bin_path = os.path.join(output_dir, "vliw.image")
    txt_path = os.path.join(output_dir, "vliw_offset.txt")

    vliw_count = 0

    with open(bin_path, 'wb') as f_bin, \
         open(txt_path, 'w', encoding='utf-8') as f_txt:

        f_txt.write("# VLIW 指令偏移地址映射\n")
        f_txt.write("-" * 60 + "\n")

        for i, layer in enumerate(all_layers):
            inp = layer['input']
            out = layer['output']

            # ====== 确定输入地址 ======
            if i == 0:
                # 第一层从 0x0 读取输入
                input_addr = 0
            else:
                # 后续层: 输入地址 = 产生该输入的层的输出地址
                if inp in output_addrs:
                    input_addr = output_addrs[inp]
                else:
                    # 找不到输入地址，说明这是一个新的外部输入（如来自maxpool/qlinearadd的输出）
                    # 需要分配一个新地址来存储这个中间结果
                    input_addr = current_addr
                    current_addr += addr_increment
                    if current_addr >= addr_pool_end:
                        current_addr = addr_pool_start
                    # 保存这个新分配的地址
                    output_addrs[inp] = input_addr
                    used_addrs.add(input_addr)
                    print(f"New input '{inp}' at 0x{input_addr:x}")

            # ====== 确定输出地址 ======
            # 检查输出是否已经被分配过地址
            if out in output_addrs:
                # 如果输出已经分配过地址，复用该地址
                output_addr = output_addrs[out]
            else:
                # 新输出，分配新地址
                # 先尝试回收空闲地址
                if free_addrs:
                    output_addr = free_addrs.pop(0)
                else:
                    # 没有空闲地址，使用新地址
                    output_addr = current_addr
                    current_addr += addr_increment
                    if current_addr >= addr_pool_end:
                        current_addr = addr_pool_start

                # 保存输出地址
                output_addrs[out] = output_addr
                used_addrs.add(output_addr)

            # ====== 追踪依赖 - 减少输入的使用计数 ======
            if inp in output_usage:
                output_usage[inp] -= 1
                # 如果输入不再被使用，回收地址
                if output_usage[inp] == 0 and inp in output_addrs:
                    freed = output_addrs[inp]
                    # 只有非初始输入的地址才回收
                    if freed != 0 and freed in used_addrs:
                        free_addrs.append(freed)
                        print(f"Recycle address 0x{freed:x} for output '{inp}'")

            # ====== 获取权重地址 ======
            weight_addr = weight_addrs.get(layer['weight'], 0)
            bias_addr = bias_addrs.get(layer['bias'], 0)

            # ====== Op 类型 ======
            op_type = layer.get('type', 'qlinearconv')

            # ====== 生成 VLIW ======
            vliw_binary = generate_vliw_binary(
                layer,
                input_addr=input_addr,
                weight_addr=weight_addr,
                bias_addr=bias_addr,
                output_addr=output_addr,
                relu_addr=0,
                op_type=op_type
            )

            f_bin.write(vliw_binary)
            f_txt.write(f"0x{output_addr - 0x10000000:08x}: layer_{i} {out} | in:{inp} in_addr:0x{input_addr:x} out_addr:0x{output_addr:x} op:{op_type}\n")

            vliw_count += 1
            print(f"Layer {i}: {out}, input=0x{input_addr:x}, output=0x{output_addr:x}")

        f_txt.write("-" * 60 + "\n")
        f_txt.write(f"# Total: {vliw_count} VLIW instructions\n")
        f_txt.write(f"# Address pool: 0x{addr_pool_start:x} ~ 0x{addr_pool_end:x}\n")

    print(f"[VLIW] {bin_path}, {vliw_count} instructions")


if __name__ == "__main__":
    extract()