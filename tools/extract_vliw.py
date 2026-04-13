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
    'qlinearglobalaveragepool': 2,  # GlobalAvgPool
}


def parse_int_attr(attrs, name, default=0):
    """解析整数属性"""
    # 匹配 0x开头的十六进制 或 十进制数字，需要单词边界避免匹配到sM_concat等
    m = re.search(rf'\b{name}\s*=\s*(0x[0-9a-fA-F]+|\d+)', attrs)
    if m:
        val = m.group(1)
        if val.startswith('0x'):
            return int(val, 16)
        return int(val)
    return default


def parse_all_layers(mlir_path, verbose=True):
    """解析所有操作层 - 按MLIR中的顺序"""
    with open(mlir_path, 'r', encoding='utf-8') as f:
        content = f.read()

    layers = []

    # 建立tensor shape映射
    tensor_shapes = {}

    # 从func signature获取输入shape
    func_match = re.search(r'func\.func\s+@main\(%[^:]+:\s*tensor<([^>]+)>', content)
    if func_match:
        shape_str = func_match.group(1)
        dims = re.findall(r'(\d+)', shape_str)
        input_shape = [int(x) for x in dims]
        tensor_shapes['input'] = input_shape
        if verbose:
            print(f"Input shape: {input_shape}")

    # 使用正则匹配所有操作，按MLIR中的顺序
    # 匹配模式: %name = "coa.op"(...) { attrs }
    all_ops_pattern = r'%(\w+)\s*=\s*"coa\.(\w+)"\(([^)]+)\)\s*\{([^}]+)\}'

    # 先建立tensor_shapes映射（从qlinearconv中获取输出shape）
    for match in re.finditer(all_ops_pattern, content):
        out_var = match.group(1)
        op_name = match.group(2)
        attrs = match.group(4)

        # 从qlinearconv获取shape信息
        if op_name == 'qlinearconv':
            R = parse_int_attr(attrs, 'R')
            C = parse_int_attr(attrs, 'C')
            M = parse_int_attr(attrs, 'M')
            N = parse_int_attr(attrs, 'N')
            if R > 0 and C > 0 and M > 0:
                tensor_shapes[out_var] = [1, M, R, C]

    # 再次遍历，按MLIR顺序解析所有操作
    for match in re.finditer(all_ops_pattern, content):
        out_var = match.group(1)
        op_name = match.group(2)
        args_str = match.group(3)
        attrs = match.group(4)

        # 解析输入参数
        in_vars = [v.strip() for v in args_str.split(',')]

        if op_name == 'qlinearconv':
            # qlinearconv: (%input, %weight, %bias)
            in_var = in_vars[0] if len(in_vars) > 0 else ''
            weight_var = in_vars[1] if len(in_vars) > 1 else ''
            bias_var = in_vars[2] if len(in_vars) > 2 else ''

            in_scale = re.search(r'in_scale\s*=\s*([\d.]+)', attrs)
            in_zp = re.search(r'in_zp\s*=\s*(-?[\d.]+)', attrs)
            w_scale = re.search(r'weight_scale\s*=\s*([\d.]+)', attrs)
            w_zp = re.search(r'weight_zp\s*=\s*(-?[\d.]+)', attrs)
            out_scale = re.search(r'out_scale\s*=\s*([\d.]+)', attrs)
            out_zp = re.search(r'out_zp\s*=\s*(-?[\d.]+)', attrs)
            dilations = re.search(r'dilations\s*=\s*\[(\d+),\s*(\d+)\]', attrs)
            kernel = re.search(r'kernel_shape\s*=\s*\[(\d+),\s*(\d+)\]', attrs)
            pads = re.search(r'pads\s*=\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)', attrs)
            strides = re.search(r'strides\s*=\s*\[(\d+),\s*(\d+)\]', attrs)

            in_addr = parse_int_attr(attrs, 'in_addr')
            out_addr = parse_int_attr(attrs, 'out_addr')
            weight_addr = parse_int_attr(attrs, 'weight_addr')
            bias_addr = parse_int_attr(attrs, 'bias_addr')
            silu_addr = parse_int_attr(attrs, 'silu_addr')

            R = parse_int_attr(attrs, 'R')
            C = parse_int_attr(attrs, 'C')
            M = parse_int_attr(attrs, 'M')
            N = parse_int_attr(attrs, 'N')
            R0 = parse_int_attr(attrs, 'R0')
            C0 = parse_int_attr(attrs, 'C0')
            sM_concat = parse_int_attr(attrs, 'sM_concat')
            M_concat = parse_int_attr(attrs, 'M_concat')
            tM = parse_int_attr(attrs, 'tM')
            tR = parse_int_attr(attrs, 'tR')
            tC = parse_int_attr(attrs, 'tC')
            factor = parse_int_attr(attrs, 'factor')

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
                'kernel': [int(kernel.group(1)), int(kernel.group(2))] if kernel else [3, 3],
                'pads': [int(pads.group(i)) for i in range(1, 5)] if pads else [0, 0, 0, 0],
                'strides': [int(strides.group(1)), int(strides.group(2))] if strides else [1, 1],
                'in_addr': in_addr,
                'out_addr': out_addr,
                'weight_addr': weight_addr,
                'bias_addr': bias_addr,
                'silu_addr': silu_addr,
                'R': R, 'C': C, 'M': M, 'N': N,
                'R0': R0, 'C0': C0,
                'sM_concat': sM_concat, 'M_concat': M_concat,
                'tM': tM, 'tR': tR, 'tC': tC,
                'factor': factor,
                'type': 'qlinearconv'
            })
            tensor_shapes[out_var] = [1, M, R, C]

        elif op_name == 'maxpool':
            # maxpool: (%input)
            in_var = in_vars[0] if len(in_vars) > 0 else ''

            kernel = re.search(r'kernel_shape\s*=\s*\[(\d+),\s*(\d+)\]', attrs)
            pads = re.search(r'pads\s*=\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)', attrs)
            strides = re.search(r'strides\s*=\s*\[(\d+),\s*(\d+)\]', attrs)

            in_addr = parse_int_attr(attrs, 'in_addr')
            out_addr = parse_int_attr(attrs, 'out_addr')

            k = int(kernel.group(1)) if kernel else 3
            p = int(pads.group(1)) if pads else 0
            s = int(strides.group(1)) if strides else 1

            # 从tensor_shapes获取输入shape并计算输出shape
            in_shape = tensor_shapes.get(in_var, [1, 64, 112, 112])
            N_in, C_in, H_in, W_in = in_shape[0], in_shape[1], in_shape[2], in_shape[3]
            H_out = (H_in + 2 * p - k) // s + 1
            W_out = (W_in + 2 * p - k) // s + 1

            layers.append({
                'output': out_var,
                'input': in_var,
                'kernel': [k, k],
                'pads': [p, p, p, p],
                'strides': [s, s],
                'in_addr': in_addr,
                'out_addr': out_addr,
                'R': H_out, 'C': W_out, 'M': C_in, 'N': C_in,
                'R0': H_in, 'C0': W_in,
                'tR': H_out, 'tC': W_out, 'tM': C_in, 'tN': C_in,
                'type': 'maxpool'
            })
            tensor_shapes[out_var] = [N_in, C_in, H_out, W_out]

        elif op_name == 'qlinearadd':
            # qlinearadd: (%input1, %input2)
            in_var1 = in_vars[0] if len(in_vars) > 0 else ''
            in_var2 = in_vars[1] if len(in_vars) > 1 else ''

            a_scale = re.search(r'a_scale\s*=\s*([\d.]+)', attrs)
            a_zp = re.search(r'a_zp\s*=\s*(-?[\d.]+)', attrs)
            b_scale = re.search(r'b_scale\s*=\s*([\d.]+)', attrs)
            b_zp = re.search(r'b_zp\s*=\s*(-?[\d.]+)', attrs)
            out_scale = re.search(r'out_scale\s*=\s*([\d.]+)', attrs)
            out_zp = re.search(r'out_zp\s*=\s*(-?[\d.]+)', attrs)

            in_addr = parse_int_attr(attrs, 'in_addr')
            in2_addr = parse_int_attr(attrs, 'in2_addr')
            out_addr = parse_int_attr(attrs, 'out_addr')
            factor = parse_int_attr(attrs, 'factor')
            factor2 = parse_int_attr(attrs, 'factor2')

            # 获取shape
            in_shape = tensor_shapes.get(in_var1, [1, 64, 56, 56])
            if len(in_shape) >= 4:
                N_in, C_in, H_in, W_in = in_shape[0], in_shape[1], in_shape[2], in_shape[3]
                R, C = H_in, W_in
                M, N = C_in, C_in
                R0, C0 = H_in, W_in
            else:
                R = C = M = N = R0 = C0 = 1

            layers.append({
                'output': out_var,
                'input': in_var1,
                'input2': in_var2,
                'a_scale': float(a_scale.group(1)) if a_scale else 1.0,
                'a_zp': int(a_zp.group(1)) if a_zp else 0,
                'b_scale': float(b_scale.group(1)) if b_scale else 1.0,
                'b_zp': int(b_zp.group(1)) if b_zp else 0,
                'out_scale': float(out_scale.group(1)) if out_scale else 1.0,
                'out_zp': int(out_zp.group(1)) if out_zp else 0,
                'in_addr': in_addr,
                'in2_addr': in2_addr,
                'out_addr': out_addr,
                'factor': factor,
                'factor2': factor2,
                'R': R, 'C': C, 'M': M, 'N': N,
                'R0': R0, 'C0': C0,
                'tR': R, 'tC': C, 'tM': M, 'tN': N,
                'type': 'qlinearadd'
            })
            tensor_shapes[out_var] = in_shape

        elif op_name == 'qgemm':
            # qgemm: (%input, %weight, %bias)
            in_var = in_vars[0] if len(in_vars) > 0 else ''
            weight_var = in_vars[1] if len(in_vars) > 1 else ''
            bias_var = in_vars[2] if len(in_vars) > 2 else ''

            a_scale = re.search(r'a_scale\s*=\s*([\d.]+)', attrs)
            a_zp = re.search(r'a_zp\s*=\s*(-?[\d.]+)', attrs)
            b_scale = re.search(r'b_scale\s*=\s*([\d.]+)', attrs)
            b_zp = re.search(r'b_zp\s*=\s*(-?[\d.]+)', attrs)
            out_scale = re.search(r'out_scale\s*=\s*([\d.]+)', attrs)
            out_zp = re.search(r'out_zp\s*=\s*(-?[\d.]+)', attrs)
            transB = re.search(r'transB\s*=\s*(\d+)', attrs)

            in_addr = parse_int_attr(attrs, 'in_addr')
            out_addr = parse_int_attr(attrs, 'out_addr')
            weight_addr = parse_int_attr(attrs, 'weight_addr')
            bias_addr = parse_int_attr(attrs, 'bias_addr')
            silu_addr = parse_int_attr(attrs, 'silu_addr')

            R = parse_int_attr(attrs, 'R')
            C = parse_int_attr(attrs, 'C')
            M = parse_int_attr(attrs, 'M')
            N = parse_int_attr(attrs, 'N')
            R0 = parse_int_attr(attrs, 'R0')
            C0 = parse_int_attr(attrs, 'C0')
            sM_concat = parse_int_attr(attrs, 'sM_concat')
            M_concat = parse_int_attr(attrs, 'M_concat')
            tM = parse_int_attr(attrs, 'tM')
            tR = parse_int_attr(attrs, 'tR')
            tC = parse_int_attr(attrs, 'tC')
            factor = parse_int_attr(attrs, 'factor')

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
                'in_addr': in_addr,
                'out_addr': out_addr,
                'weight_addr': weight_addr,
                'bias_addr': bias_addr,
                'silu_addr': silu_addr,
                'R': R, 'C': C, 'M': M, 'N': N,
                'R0': R0, 'C0': C0,
                'sM_concat': sM_concat, 'M_concat': M_concat,
                'tM': tM, 'tR': tR, 'tC': tC,
                'factor': factor,
                'type': 'qgemm'
            })

        elif op_name == 'qlinearglobalaveragepool':
            # qlinearglobalaveragepool: (%input)
            in_var = in_vars[0] if len(in_vars) > 0 else ''

            in_scale = re.search(r'in_scale\s*=\s*([\d.]+)', attrs)
            in_zp = re.search(r'in_zp\s*=\s*(-?[\d.]+)', attrs)
            out_scale = re.search(r'out_scale\s*=\s*([\d.]+)', attrs)
            out_zp = re.search(r'out_zp\s*=\s*(-?[\d.]+)', attrs)

            in_addr = parse_int_attr(attrs, 'in_addr')
            out_addr = parse_int_attr(attrs, 'out_addr')

            # 获取shape
            in_shape = tensor_shapes.get(in_var, [1, 512, 7, 7])
            if len(in_shape) >= 4:
                N_in, C_in, H_in, W_in = in_shape[0], in_shape[1], in_shape[2], in_shape[3]
                R, C = 1, 1
                M, N = C_in, C_in
                R0, C0 = H_in, W_in
            else:
                R = C = M = N = R0 = C0 = 1

            layers.append({
                'output': out_var,
                'input': in_var,
                'in_scale': float(in_scale.group(1)) if in_scale else 1.0,
                'in_zp': int(in_zp.group(1)) if in_zp else 0,
                'out_scale': float(out_scale.group(1)) if out_scale else 1.0,
                'out_zp': int(out_zp.group(1)) if out_zp else 0,
                'in_addr': in_addr,
                'out_addr': out_addr,
                'R': R, 'C': C, 'M': M, 'N': N,
                'R0': R0, 'C0': C0,
                'tR': R, 'tC': C, 'tM': M, 'tN': N,
                'type': 'qlinearglobalaveragepool'
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


def get_quant_factor(in_scale, w_scale, out_scale, bits=36):
    """计算量化因子 - 完全模拟C代码"""
    import numpy as np
    import struct
    # C代码: factor = (float)(*sx)/(float)(*so)*(float)(*sw)
    # 每次运算都转换为float (单精度)
    tmp = np.float32(in_scale) / np.float32(out_scale)
    factor = tmp * np.float32(w_scale)
    # 转换为Python float进行后续计算
    factor_float = float(factor)
    # 模拟C中float->double转换
    packed = struct.pack('f', factor_float)
    factor_converted = struct.unpack('f', packed)[0]
    return int(factor_converted * (1 << bits))


def output_vliw_hex(fp_hex, vliw_dict):
    """将VLIW字段以16进制格式输出到txt文件 (与C代码格式一致)"""
    if fp_hex is None:
        return

    fp_hex.write(f"operator=0x{vliw_dict['operator'] & 0xFF:02x}\n")
    fp_hex.write(f"DDR_x1_address=0x{vliw_dict['DDR_x1_address'] & 0xFFFFFFFFF:09x}\n")
    fp_hex.write(f"DDR_x2_address=0x{vliw_dict['DDR_x2_address'] & 0xFFFFFFFFF:09x}\n")
    fp_hex.write(f"Bias_source_address=0x{vliw_dict['Bias_source_address'] & 0x7FF:03x}\n")
    fp_hex.write(f"Compute_Result_dest_address=0x{vliw_dict['Compute_Result_dest_address'] & 0xFFFFFFFFF:09x}\n")
    fp_hex.write(f"Activate_LUT_address=0x{vliw_dict['Activate_LUT_address'] & 0xFF:02x}\n")
    fp_hex.write(f"R=0x{vliw_dict['R'] & 0x7FF:03x}\n")
    fp_hex.write(f"C=0x{vliw_dict['C'] & 0x7FF:03x}\n")
    fp_hex.write(f"M=0x{vliw_dict['M'] & 0xFFF:03x}\n")
    fp_hex.write(f"N=0x{vliw_dict['N'] & 0xFFF:03x}\n")
    fp_hex.write(f"R0=0x{vliw_dict['R0'] & 0x7FF:03x}\n")
    fp_hex.write(f"C0=0x{vliw_dict['C0'] & 0x7FF:03x}\n")
    fp_hex.write(f"sM_concat=0x{vliw_dict['sM_concat'] & 0xFFF:03x}\n")
    fp_hex.write(f"M_concat=0x{vliw_dict['M_concat'] & 0xFFF:03x}\n")
    fp_hex.write(f"Quant_x1_z=0x{vliw_dict['Quant_x1_z'] & 0xFF:02x}\n")
    fp_hex.write(f"Quant_x2_z=0x{vliw_dict['Quant_x2_z'] & 0xFF:02x}\n")
    fp_hex.write(f"Quant_y_z=0x{vliw_dict['Quant_y_z'] & 0xFF:02x}\n")
    fp_hex.write(f"Conv_pad=0x{vliw_dict['Conv_pad'] & 0x7:01x}\n")
    fp_hex.write(f"Conv_kernel=0x{vliw_dict['Conv_kernel'] & 0x1F:02x}\n")
    fp_hex.write(f"Conv_stride=0x{vliw_dict['Conv_stride'] & 0x7:01x}\n")
    fp_hex.write(f"Conv_dilation=0x{vliw_dict['Conv_dilation'] & 0x7:01x}\n")
    fp_hex.write(f"Conv_tR=0x{vliw_dict['Conv_tR'] & 0x7FF:03x}\n")
    fp_hex.write(f"Conv_tC=0x{vliw_dict['Conv_tC'] & 0x7FF:03x}\n")
    fp_hex.write(f"Conv_tM=0x{vliw_dict['Conv_tM'] & 0xFFF:03x}\n")
    fp_hex.write(f"Conv_tN=0x{vliw_dict['Conv_tN'] & 0xFFF:03x}\n")
    fp_hex.write(f"Conv_permuteR=0x{vliw_dict['Conv_permuteR'] & 0x3:01x}\n")
    fp_hex.write(f"Conv_permuteC=0x{vliw_dict['Conv_permuteC'] & 0x3:01x}\n")
    fp_hex.write(f"Conv_permuteM=0x{vliw_dict['Conv_permuteM'] & 0x3:01x}\n")
    fp_hex.write(f"Conv_permuteN=0x{vliw_dict['Conv_permuteN'] & 0x3:01x}\n")
    fp_hex.write(f"Conv_quant_factor=0x{vliw_dict['Conv_quant_factor'] & 0x3FFFFFFFF:06x}\n")
    fp_hex.write(f"Conv_quant_factor2=0x{vliw_dict['Conv_quant_factor2'] & 0xFFFFFFFF:06x}\n")
    fp_hex.write("---\n")


def generate_vliw_dict(layer_info, op_type='qlinearconv'):
    """生成 VLIW 字典 - 直接从layer_info中读取参数"""
    # 获取 operator 类型
    operator = OperatorMap.get(op_type, 0)

    if op_type == 'maxpool':
        # MaxPool: operator=1
        return {
            'operator': operator,
            'DDR_x1_address': layer_info.get('in_addr', 0),
            'DDR_x2_address': 0,
            'Bias_source_address': 0,
            'Compute_Result_dest_address': layer_info.get('out_addr', 0),
            'Activate_LUT_address': 0,
            'R': layer_info.get('R', 1),
            'C': layer_info.get('C', 1),
            'M': layer_info.get('M', 1),
            'N': layer_info.get('N', 1),
            'R0': layer_info.get('R0', 1),
            'C0': layer_info.get('C0', 1),
            'sM_concat': 0,
            'M_concat': 0,
            'Quant_x1_z': -128,
            'Quant_x2_z': 0,
            'Quant_y_z': 0,
            'Conv_pad': layer_info.get('pads', [0,0,0,0])[0],
            'Conv_kernel': layer_info.get('kernel', [1,1])[0],
            'Conv_stride': layer_info.get('strides', [1,1])[0],
            'Conv_dilation': 1,
            'Conv_tR': layer_info.get('tR', 1),
            'Conv_tC': layer_info.get('tC', 1),
            'Conv_tM': layer_info.get('M', 1),
            'Conv_tN': layer_info.get('N', 1),
            'Conv_permuteR': 0,
            'Conv_permuteC': 0,
            'Conv_permuteM': 0,
            'Conv_permuteN': 0,
            'Conv_quant_factor': 0,
            'Conv_quant_factor2': 0,
        }

    elif op_type == 'qlinearadd':
        # Add: operator=3
        # factor = a_scale * a_zp / out_scale
        factor = layer_info.get('factor', 0)
        factor2 = layer_info.get('factor2', 0)
        return {
            'operator': operator,
            'DDR_x1_address': layer_info.get('in_addr', 0),
            'DDR_x2_address': layer_info.get('in2_addr', 0),
            'Bias_source_address': 0,
            'Compute_Result_dest_address': layer_info.get('out_addr', 0),
            'Activate_LUT_address': 0,
            'R': layer_info.get('R', 1),
            'C': layer_info.get('C', 1),
            'M': layer_info.get('M', 1),
            'N': layer_info.get('N', 1),
            'R0': layer_info.get('R0', 1),
            'C0': layer_info.get('C0', 1),
            'sM_concat': 0,
            'M_concat': 0,
            'Quant_x1_z': layer_info.get('a_zp', 0),
            'Quant_x2_z': layer_info.get('b_zp', 0),
            'Quant_y_z': layer_info.get('out_zp', 0),
            'Conv_pad': 0,
            'Conv_kernel': 1,
            'Conv_stride': 1,
            'Conv_dilation': 1,
            'Conv_tR': layer_info.get('tR', 1),
            'Conv_tC': layer_info.get('tC', 1),
            'Conv_tM': layer_info.get('M', 1),
            'Conv_tN': layer_info.get('N', 1),
            'Conv_permuteR': 0,
            'Conv_permuteC': 0,
            'Conv_permuteM': 0,
            'Conv_permuteN': 0,
            'Conv_quant_factor': factor,
            'Conv_quant_factor2': factor2,
        }

    elif op_type == 'qlinearglobalaveragepool':
        # GlobalAvgPool: operator=2 (上采样模式)
        return {
            'operator': operator,
            'DDR_x1_address': layer_info.get('in_addr', 0),
            'DDR_x2_address': 0,
            'Bias_source_address': 0,
            'Compute_Result_dest_address': layer_info.get('out_addr', 0),
            'Activate_LUT_address': 0,
            'R': layer_info.get('R', 1),
            'C': layer_info.get('C', 1),
            'M': layer_info.get('M', 1),
            'N': layer_info.get('N', 1),
            'R0': layer_info.get('R0', 1),
            'C0': layer_info.get('C0', 1),
            'sM_concat': 0,
            'M_concat': 0,
            'Quant_x1_z': -128,
            'Quant_x2_z': 0,
            'Quant_y_z': layer_info.get('out_zp', 0),
            'Conv_pad': 0,
            'Conv_kernel': 1,
            'Conv_stride': 1,
            'Conv_dilation': 1,
            'Conv_tR': 1,
            'Conv_tC': 1,
            'Conv_tM': layer_info.get('M', 1),
            'Conv_tN': layer_info.get('N', 1),
            'Conv_permuteR': 0,
            'Conv_permuteC': 0,
            'Conv_permuteM': 0,
            'Conv_permuteN': 0,
            'Conv_quant_factor': 0,
            'Conv_quant_factor2': 0,
        }

    else:
        # qlinearconv / qgemm: operator=0
        # 使用get_quant_factor重新计算（模拟C代码）
        qf = get_quant_factor(
            layer_info.get('in_scale', 1.0),
            layer_info.get('w_scale', 1.0),
            layer_info.get('out_scale', 1.0),
            bits=36
        )

        return {
            'operator': operator,
            'DDR_x1_address': layer_info.get('in_addr', 0),
            'DDR_x2_address': layer_info.get('weight_addr', 0),
            'Bias_source_address': layer_info.get('bias_addr', 0),
            'Compute_Result_dest_address': layer_info.get('out_addr', 0),
            'Activate_LUT_address': layer_info.get('silu_addr', 0),
            'R': layer_info.get('R', 1),
            'C': layer_info.get('C', 1),
            'M': layer_info.get('M', 1),
            'N': layer_info.get('N', 1),
            'R0': layer_info.get('R0', 1),
            'C0': layer_info.get('C0', 1),
            'sM_concat': layer_info.get('sM_concat', 0),
            'M_concat': layer_info.get('M_concat', 0),
            'Quant_x1_z': layer_info.get('in_zp', 0),
            'Quant_x2_z': layer_info.get('w_zp', 0),
            'Quant_y_z': layer_info.get('out_zp', 0),
            'Conv_pad': layer_info.get('pads', [0,0,0,0])[0],
            'Conv_kernel': layer_info.get('kernel', [3,3])[0],
            'Conv_stride': layer_info.get('strides', [1,1])[0],
            'Conv_dilation': layer_info.get('dilations', [1,1])[0],
            'Conv_tR': layer_info.get('tR', 1),
            'Conv_tC': layer_info.get('tC', 1),
            'Conv_tM': layer_info.get('tM', 1),
            'Conv_tN': layer_info.get('N', 1),
            'Conv_permuteR': 0,
            'Conv_permuteC': 0,
            'Conv_permuteM': 0,
            'Conv_permuteN': 0,
            'Conv_quant_factor': qf,
            'Conv_quant_factor2': 0,
        }


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
    qf = get_quant_factor(
        layer_info.get('in_scale', 1.0),
        layer_info.get('w_scale', 1.0),
        layer_info.get('out_scale', 1.0),
        bits=36
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


def extract(verbose=True):
    """生成 VLIW 指令 - 使用带addr的MLIR文件

    Args:
        verbose: 是否打印详细信息到终端
    """
    # 使用带地址的MLIR文件
    mlir_path = os.path.join(PROJECT_ROOT, "models", "mlir_models", "resnet18_quant_int8_addr.mlir")
    output_dir = os.path.join(PROJECT_ROOT, "parameters")

    # 解析所有层信息 - 按MLIR中的顺序
    all_layers = parse_all_layers(mlir_path, verbose=verbose)

    if verbose:
        print(f"Found {len(all_layers)} layers")

    # 输出文件
    bin_path = os.path.join(output_dir, "vliw.image")
    txt_path = os.path.join(output_dir, "vliw_offset.txt")
    hex_path = os.path.join(output_dir, "vliw_hex.txt")

    vliw_count = 0

    with open(bin_path, 'wb') as f_bin, \
         open(txt_path, 'w', encoding='utf-8') as f_txt, \
         open(hex_path, 'w', encoding='utf-8') as f_hex:

        f_txt.write("# VLIW 指令偏移地址映射 (from MLIR with addr)\n")
        f_txt.write("-" * 60 + "\n")

        for i, layer in enumerate(all_layers):
            out = layer['output']
            op_type = layer.get('type', 'qlinearconv')

            # 直接从MLIR中读取地址和参数
            vliw_dict = generate_vliw_dict(layer, op_type)
            output_vliw_hex(f_hex, vliw_dict)

            # 生成占位符二进制 (64 bytes)
            vliw_binary = bytes(64)
            f_bin.write(vliw_binary)

            # 记录信息
            in_addr = layer.get('in_addr', 0)
            out_addr = layer.get('out_addr', 0)
            f_txt.write(f"Layer {i}: {out} | op:{op_type} in_addr:0x{in_addr:x} out_addr:0x{out_addr:x}\n")

            vliw_count += 1
            if verbose:
                print(f"Layer {i}: {out}, op={op_type}, in=0x{in_addr:x}, out=0x{out_addr:x}")

        f_txt.write("-" * 60 + "\n")
        f_txt.write(f"# Total: {vliw_count} VLIW instructions\n")

    if verbose:
        print(f"[VLIW] {bin_path}, {vliw_count} instructions")
        print(f"[HEX] {hex_path}")


def test():
    """测试函数 - 验证生成的指令"""
    import os

    py_hex = os.path.join(PROJECT_ROOT, "parameters", "vliw_hex.txt")
    c_hex = os.path.join(PROJECT_ROOT, "c_reference", "instruction", "instruction_hex.txt")

    print("=" * 60)
    print("测试: 比较 Python 和 C 代码生成的 VLIW 指令")
    print("=" * 60)

    # 读取 Python 输出
    with open(py_hex, 'r') as f:
        py_lines = f.readlines()

    # 读取 C 代码输出
    with open(c_hex, 'r') as f:
        c_lines = f.readlines()

    # 解析各指令块
    def parse_blocks(lines):
        blocks = []
        block = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line == '---':
                if block:
                    blocks.append(block)
                    block = {}
            elif '=' in line:
                key, val = line.split('=', 1)
                block[key] = val
        if block:
            blocks.append(block)
        return blocks

    py_blocks = parse_blocks(py_lines)
    c_blocks = parse_blocks(c_lines)

    print(f"Python 生成了 {len(py_blocks)} 条指令")
    print(f"C 代码生成了 {len(c_blocks)} 条指令")

    # 关键字段列表
    key_fields = ['operator', 'R', 'C', 'M', 'N', 'R0', 'C0', 'sM_concat', 'M_concat']

    mismatches = []
    for i, (py_b, c_b) in enumerate(zip(py_blocks, c_blocks)):
        for field in key_fields:
            py_val = py_b.get(field, '0')
            c_val = c_b.get(field, '0')
            if py_val != c_val:
                mismatches.append(f"Layer {i}: {field} Python={py_val} C={c_val}")

    if mismatches:
        print(f"\n发现 {len(mismatches)} 个不匹配:")
        for m in mismatches[:10]:  # 只显示前10个
            print(f"  - {m}")
        if len(mismatches) > 10:
            print(f"  ... 还有 {len(mismatches) - 10} 个")
    else:
        print("\n✓ 所有关键字段匹配!")

    # 测试 maxpool 和 qlinearadd
    print("\n" + "=" * 60)
    print("测试: maxpool (operator=0x01)")
    print("=" * 60)

    for i, b in enumerate(py_blocks):
        if b.get('operator') == '0x01':
            print(f"Layer {i}: R={b.get('R')}, C={b.get('C')}, M={b.get('M')}, N={b.get('N')}")
            print(f"         R0={b.get('R0')}, C0={b.get('C0')}")
            print(f"         kernel={b.get('Conv_kernel')}, stride={b.get('Conv_stride')}, pad={b.get('Conv_pad')}")

    print("\n" + "=" * 60)
    print("测试: qlinearadd (operator=0x03)")
    print("=" * 60)

    add_count = 0
    for i, b in enumerate(py_blocks):
        if b.get('operator') == '0x03':
            add_count += 1
            print(f"Layer {i}: R={b.get('R')}, C={b.get('C')}, M={b.get('M')}, N={b.get('N')}")

    print(f"\n共 {add_count} 个 qlinearadd 层")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='生成 VLIW 指令')
    parser.add_argument('--verbose', '-v', action='store_true', help='打印详细信息')
    parser.add_argument('--test', '-t', action='store_true', help='运行测试')
    args = parser.parse_args()

    if args.test:
        test()
    else:
        extract(verbose=args.verbose)