"""
地址分配 - 带基地址偏移
- 第一层输入: 0x0
- 其他输入/输出: +0x10000000
- Weight: 0x8000000 (IC padded to 32)
- Bias: 0xC0000000
"""

import os
import re
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def calc_tensor_size(shape):
    size = 1
    for d in shape:
        size *= d
    return size


def calculate_buffer_consumption(tN, tM, tR, tC, N, M, kernel, stride, pad, dilation):
    """计算buffer使用情况，返回0表示OK"""
    # tN32_rd = ceil(tN/32) 的数量
    tN32_rd = 0
    for sn in range(0, N, tN):
        t = ((sn + tN + 31) >> 5) - (sn >> 5)
        if t > tN32_rd:
            tN32_rd = t

    tM32_rd = 0
    for sm in range(0, M, tM):
        t = ((sm + tM + 15) >> 4) - (sm >> 4)
        if t > tM32_rd:
            tM32_rd = t

    relems = (tR - 1) * stride + (kernel - 1) * dilation + 1
    celems = (tC - 1) * stride + (kernel - 1) * dilation + 1
    gdepth = relems * celems * tN32_rd
    wbuf_single_m_size = tN32_rd * kernel * kernel
    wdepth = wbuf_single_m_size * tM32_rd
    odepth = tR * tC * tM32_rd

    if wdepth >= 256:
        return 1
    if gdepth >= 1024:
        return 2
    if odepth >= 2048:
        return 3
    return 0


def get_tile(N, M, R, C, kernel, stride, pad, dilation):
    """计算tile大小 tM, tR, tC"""
    tM = M
    tR = R
    tC = C

    flag = calculate_buffer_consumption(N, tM, tR, tC, N, M, kernel, stride, pad, dilation)
    while flag != 0:
        if flag == 1:
            # wdepth 超限，减少 tM
            if tM % 2 == 0:
                tM = tM // 2
            elif tM % 3 == 0:
                tM = tM // 3
            elif tM % 4 == 0:
                tM = tM // 4
            elif tM % 5 == 0:
                tM = tM // 5
            else:
                tM = 16
            if tM < 16:
                tM = 16
        if flag == 2 or flag == 3:
            # gdepth 或 odepth 超限，减少 tR/tC
            if tR != 1:
                if tR % 2 == 0:
                    tR = tR // 2
                elif tR % 3 == 0:
                    tR = tR // 3
                elif tR % 4 == 0:
                    tR = tR // 4
                elif tR % 5 == 0:
                    tR = tR // 5
                elif tR % 6 == 0:
                    tR = tR // 6
                else:
                    tR = 1
            else:
                if tC % 2 == 0:
                    tC = tC // 2
                elif tC % 3 == 0:
                    tC = tC // 3
                elif tC % 4 == 0:
                    tC = tC // 4
                elif tC % 5 == 0:
                    tC = tC // 5
                elif tC % 6 == 0:
                    tC = tC // 6
                else:
                    tC = 1

        flag = calculate_buffer_consumption(N, tM, tR, tC, N, M, kernel, stride, pad, dilation)
        if flag == 0:
            # 尝试增大
            tR_old = tR
            tR = tR * 2
            flag = calculate_buffer_consumption(N, tM, tR, tC, N, M, kernel, stride, pad, dilation)
            if flag == 0:
                tR = tR * 2
                flag = calculate_buffer_consumption(N, tM, tR, tC, N, M, kernel, stride, pad, dilation)
                if flag != 0:
                    tR = tR // 2
                    flag = calculate_buffer_consumption(N, tM, tR, tC, N, M, kernel, stride, pad, dilation)
            else:
                tR = tR_old

    return tM, tR, tC


def assign_addresses(mlir_path, output_path):
    with open(mlir_path, 'r') as f:
        mlir_content = f.read()

    lines = mlir_content.split('\n')
    new_lines = []

    BASE_ADDR = 0x0
    BASE_OFFSET = 0x10000000  # 基地址偏移

    shape_estimate = {
        'input': (1, 3, 224, 224),
        '192': (1, 64, 112, 112), '126': (1, 64, 56, 56), '195': (1, 64, 56, 56),
        '198': (1, 64, 56, 56), '132': (1, 64, 56, 56), '201': (1, 64, 56, 56),
        '204': (1, 64, 56, 56), '139': (1, 64, 56, 56), '207': (1, 128, 28, 28),
        '210': (1, 128, 28, 28), '213': (1, 128, 28, 28), '148': (1, 128, 28, 28),
        '216': (1, 128, 28, 28), '219': (1, 128, 28, 28), '155': (1, 128, 28, 28),
        '222': (1, 256, 14, 14), '225': (1, 256, 14, 14), '228': (1, 256, 14, 14),
        '164': (1, 256, 14, 14), '231': (1, 256, 14, 14), '234': (1, 256, 14, 14),
        '171': (1, 256, 14, 14), '237': (1, 512, 7, 7), '240': (1, 512, 7, 7),
        '243': (1, 512, 7, 7), '180': (1, 512, 7, 7), '246': (1, 512, 7, 7),
        '249': (1, 512, 7, 7), '187': (1, 512, 7, 7), '189': (1, 512, 1, 1),
        '190': (1, 512), 'output': (1, 1000),
    }

    # 第一个操作标志
    first_op = True

    tensor_addr = {'input': BASE_ADDR}

    # ========== 计算 weight 地址 ==========
    WEIGHT_BASE = 0x8000000
    BIAS_BASE = 0xC0000000
    SILU_BASE = 0x10000000  # silu LUT 基地址，每层 256 bytes

    npz_path = os.path.join(PROJECT_ROOT, "models", "intermediate", "resnet18_quant_int8.npz")
    npz = np.load(npz_path, allow_pickle=True)

    weight_addr = WEIGHT_BASE
    bias_addr = BIAS_BASE
    silu_addr = SILU_BASE
    weight_info = {}  # (weight_addr, bias_addr, weight_size, bias_size)
    silu_info = {}  # (silu_addr, silu_size)
    tile_info = {}  # (tM, tR, tC)

    # 提取所有 qlinearconv 的权重和参数
    pattern = r'%(\w+)\s*=\s*"coa\.qlinearconv"\(%[^,]+,\s*%(\w+),\s*%(\w+)\)\s*\{([^}]+)\}'
    conv_matches = re.findall(pattern, mlir_content)

    for out_var, weight_var, bias_var, params_str in conv_matches:
        if weight_var in npz.files:
            w = npz[weight_var]
            b = npz[bias_var]

            # 计算 weight size: OC * H * W * padded_IC
            oc, ic, h, w_sz = w.shape
            ic_padded = ((ic + 31) // 32) * 32
            w_size = oc * h * w_sz * ic_padded
            b_size = len(b) * 4

            weight_info[weight_var] = (weight_addr, bias_addr, w_size, b_size)
            weight_addr += w_size
            bias_addr += b_size

            # 提取 stride, pad, dilation
            stride = 1
            pad = 0
            dilation = 1
            kernel = h  # assume square kernel

            s_match = re.search(r'strides = \[(\d+),', params_str)
            if s_match:
                stride = int(s_match.group(1))
            p_match = re.search(r'pads = \[(\d+),', params_str)
            if p_match:
                pad = int(p_match.group(1))
            d_match = re.search(r'dilations = \[(\d+),', params_str)
            if d_match:
                dilation = int(d_match.group(1))
            k_match = re.search(r'kernel_shape = \[(\d+),', params_str)
            if k_match:
                kernel = int(k_match.group(1))

            # 计算输出尺寸
            out_shape = shape_estimate.get(out_var.replace('_quantized', ''), (1, 1, 1, 1))
            out_H = out_shape[2] if len(out_shape) >= 4 else 1

            # 计算 tile
            N = ic  # input channels
            M = oc  # output channels
            R = out_H  # output height
            C = out_shape[3] if len(out_shape) >= 4 else 1  # output width

            tM, tR, tC = get_tile(N, M, R, C, kernel, stride, pad, dilation)
            tile_info[out_var] = (tM, tR, tC)

    # FC 层
    if 'fc.weight_quantized' in npz.files:
        w = npz['fc.weight_quantized']
        b = npz['fc.bias_quantized']
        in_dim, out_dim = w.shape
        in_padded = ((in_dim + 31) // 32) * 32
        w_size = in_padded * out_dim
        b_size = out_dim * 4
        weight_info['fc.weight_quantized'] = (weight_addr, bias_addr, w_size, b_size)
        weight_info['fc_weight_quantized'] = (weight_addr, bias_addr, w_size, b_size)  # qgemm 用这个键名

    print("=" * 60)
    print(f"Weight base: 0x{WEIGHT_BASE:x}, Bias base: 0x{BIAS_BASE:x}")
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            new_lines.append("")
            continue

        op_type = None
        if '"coa.qlinearconv"' in line_stripped:
            op_type = 'qlinearconv'
        elif '"coa.maxpool"' in line_stripped:
            op_type = 'maxpool'
        elif '"coa.qlinearadd"' in line_stripped:
            op_type = 'qlinearadd'
        elif '"coa.qlinearglobalaveragepool"' in line_stripped:
            op_type = 'qlinearglobalaveragepool'
        elif '"coa.qgemm"' in line_stripped:
            op_type = 'qgemm'
        else:
            new_lines.append(line_stripped)
            continue

        if op_type:
            output_match = re.search(r'%(\w+)\s*=\s*"coa\.\w+"', line_stripped)
            if output_match:
                output_name = output_match.group(1)
                base_name = output_name.replace('_quantized', '')
                out_size = calc_tensor_size(shape_estimate.get(base_name, (1,)))

                # 找输入
                all_inputs = re.findall(r'%(\w+_quantized)', line_stripped)
                inputs = [i for i in all_inputs if i.replace('_quantized', '') != base_name]

                if not inputs:
                    inputs = re.findall(r'%\w+,', line_stripped)
                    inputs = [i.rstrip(',') + '_quantized' for i in inputs]

                if op_type == 'qlinearadd':
                    in_addrs = [tensor_addr.get(i.replace('_quantized', ''), BASE_ADDR) for i in inputs]
                    in_addr, in_size = in_addrs[0], out_size
                    in2_addr = in_addrs[1] if len(in_addrs) > 1 else 0
                    in2_size = out_size

                    out_addr = in_addr + in_size
                    tensor_addr[base_name] = out_addr

                    # 应用基地址偏移
                    in_addr_final = in_addr + BASE_OFFSET
                    in2_addr_final = in2_addr + BASE_OFFSET
                    out_addr_final = out_addr + BASE_OFFSET

                    # 提取 scale 值并计算 factor (qlinearadd 有两个 factor)
                    a_scale = 1.0
                    b_scale = 1.0
                    out_scale = 1.0

                    a_match = re.search(r'a_scale\s*=\s*([0-9.]+)', line_stripped)
                    if a_match:
                        a_scale = float(a_match.group(1))
                    b_match = re.search(r'b_scale\s*=\s*([0-9.]+)', line_stripped)
                    if b_match:
                        b_scale = float(b_match.group(1))
                    out_match = re.search(r'out_scale\s*=\s*([0-9.]+)', line_stripped)
                    if out_match:
                        out_scale = float(out_match.group(1))

                    # 计算 factor: a_scale / out_scale * 2^28, b_scale / out_scale * 2^28
                    factor = int(a_scale / out_scale * (2 ** 28))
                    factor2 = int(b_scale / out_scale * (2 ** 28))

                    addr_attrs = f", in_addr = 0x{in_addr_final:x}, in_size = {in_size}, in2_addr = 0x{in2_addr_final:x}, in2_size = {in2_size}, out_addr = 0x{out_addr_final:x}, out_size = {out_size}, factor = {factor}, factor2 = {factor2}"
                elif op_type in ('qlinearconv', 'qgemm'):
                    input_key = inputs[0].replace('_quantized', '') if inputs else 'input'
                    in_addr = tensor_addr.get(input_key, BASE_ADDR)
                    in_size = out_size

                    out_addr = in_addr + in_size
                    tensor_addr[base_name] = out_addr

                    # 第一层输入用0x0，其他都用BASE_OFFSET
                    if first_op:
                        in_addr_final = in_addr  # 0x0
                        out_addr_final = out_addr + BASE_OFFSET
                    else:
                        in_addr_final = in_addr + BASE_OFFSET
                        out_addr_final = out_addr + BASE_OFFSET

                    # 查找权重地址
                    weight_var = inputs[1] if len(inputs) > 1 else None
                    w_info = weight_info.get(weight_var) if weight_var else None
                    if w_info:
                        w_addr, b_addr, w_sz, b_sz = w_info
                        # 计算 silu 地址 (每层 256 bytes)
                        s_addr = silu_info.get(output_name, (silu_addr, 256))[0]
                        silu_info[output_name] = (s_addr, 256)
                        silu_addr += 256

                        # 计算 R, C, M, N, R0, C0
                        out_shape = shape_estimate.get(base_name, (1, 1, 1, 1))
                        if len(out_shape) >= 4:
                            out_N, out_C, out_H, out_W = out_shape[0], out_shape[1], out_shape[2], out_shape[3]
                            R, C, M = out_H, out_W, out_C
                        else:
                            R, C, M = 1, 1, out_shape[1] if len(out_shape) > 1 else 1
                        M_padded = ((M + 15) // 16) * 16  # M pad to 16

                        # 获取输入形状
                        input_key = inputs[0].replace('_quantized', '') if inputs else 'input'
                        in_shape = shape_estimate.get(input_key, (1, 1, 1, 1))
                        if len(in_shape) >= 4:
                            in_N, in_C, in_H, in_W = in_shape[0], in_shape[1], in_shape[2], in_shape[3]
                            R0, C0, N = in_H, in_W, in_C
                        else:
                            R0, C0, N = 1, 1, in_shape[1] if len(in_shape) > 1 else 1
                        N_padded = ((N + 31) // 32) * 32  # N pad to 32

                        # 获取 tile 信息
                        tM, tR, tC = tile_info.get(output_name, (M_padded, R, C))

                        # 提取 scale 值并计算 factor
                        in_scale = 1.0
                        weight_scale = 1.0
                        out_scale = 1.0

                        in_match = re.search(r'in_scale\s*=\s*([0-9.]+)', line_stripped)
                        if in_match:
                            in_scale = float(in_match.group(1))
                        w_scale_match = re.search(r'weight_scale\s*=\s*([0-9.]+)', line_stripped)
                        if w_scale_match:
                            weight_scale = float(w_scale_match.group(1))
                        out_match = re.search(r'out_scale\s*=\s*([0-9.]+)', line_stripped)
                        if out_match:
                            out_scale = float(out_match.group(1))

                        # 计算 factor: in_scale * weight_scale / out_scale * 2^36
                        factor = int(in_scale * weight_scale / out_scale * (2 ** 36))

                        addr_attrs = f", in_addr = 0x{in_addr_final:x}, in_size = {in_size}, out_addr = 0x{out_addr_final:x}, out_size = {out_size}, weight_addr = 0x{w_addr:x}, weight_size = {w_sz}, bias_addr = 0x{b_addr:x}, bias_size = {b_sz}, silu_addr = 0x{s_addr:x}, silu_size = 256, R = {R}, C = {C}, M = {M_padded}, N = {N_padded}, R0 = {R0}, C0 = {C0}, sM_concat = 0, M_concat = {M_padded}, tM = {tM}, tR = {tR}, tC = {tC}, factor = {factor}"
                    else:
                        addr_attrs = f", in_addr = 0x{in_addr_final:x}, in_size = {in_size}, out_addr = 0x{out_addr_final:x}, out_size = {out_size}"
                else:
                    input_key = inputs[0].replace('_quantized', '') if inputs else 'input'
                    in_addr = tensor_addr.get(input_key, BASE_ADDR)
                    in_size = out_size

                    out_addr = in_addr + in_size
                    tensor_addr[base_name] = out_addr

                    # 第一层输入用0x0，其他都用BASE_OFFSET
                    if first_op:
                        in_addr_final = in_addr  # 0x0
                        out_addr_final = out_addr + BASE_OFFSET
                    else:
                        in_addr_final = in_addr + BASE_OFFSET
                        out_addr_final = out_addr + BASE_OFFSET

                    addr_attrs = f", in_addr = 0x{in_addr_final:x}, in_size = {in_size}, out_addr = 0x{out_addr_final:x}, out_size = {out_size}"

                if '{' in line_stripped:
                    new_line = line_stripped.replace('}', addr_attrs + ' }')
                else:
                    new_line = line_stripped + ' { ' + addr_attrs[2:] + ' }'

                print(f"  {op_type}: {output_name} in=0x{in_addr_final:x} out=0x{out_addr_final:x}")
                new_lines.append(new_line)

                first_op = False
            else:
                new_lines.append(line_stripped)
        else:
            new_lines.append(line_stripped)

    with open(output_path, 'w') as f:
        f.write('\n'.join(new_lines))
    print(f"\n[Done]")


def main():
    mlir_path = os.path.join(PROJECT_ROOT, "models", "mlir_models", "resnet18_quant_int8.mlir")
    output_path = os.path.join(PROJECT_ROOT, "models", "mlir_models", "resnet18_quant_int8_addr.mlir")
    assign_addresses(mlir_path, output_path)


if __name__ == "__main__":
    main()