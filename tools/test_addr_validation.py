"""
验证 VLIW 指令地址的正确性
1. 检查地址是否有重叠/冲突
2. 检查是否按照 MLIR 顺序执行
3. 检查数据依赖：某个算子的输出在被其他算子使用前不应该被覆盖
"""

import re
import sys
import os

def parse_vliw_hex(hex_path):
    """解析 vliw_hex.txt 文件"""
    with open(hex_path, 'r') as f:
        lines = f.readlines()

    layers = []
    current_layer = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line == '---':
            if current_layer:
                layers.append(current_layer)
                current_layer = {}
            continue

        if '=' in line:
            key, value = line.split('=', 1)
            current_layer[key] = value

    if current_layer:
        layers.append(current_layer)

    return layers


def parse_addr(addr_str):
    """解析地址字符串，返回整数"""
    if addr_str.startswith('0x'):
        return int(addr_str, 16)
    return int(addr_str)


def check_address_conflicts(layers):
    """检查地址是否有冲突（重叠）"""
    print("=" * 60)
    print("1. 检查地址冲突")
    print("=" * 60)

    # 为每个算子构建地址区间
    # DDR_x1_address: 输入地址
    # DDR_x2_address: 权重地址
    # Compute_Result_dest_address: 输出地址

    addr_ranges = []
    for i, layer in enumerate(layers):
        op_type = layer.get('operator', 'unknown')
        in_addr = parse_addr(layer.get('DDR_x1_address', '0'))
        weight_addr = parse_addr(layer.get('DDR_x2_address', '0'))
        out_addr = parse_addr(layer.get('Compute_Result_dest_address', '0'))

        # 输入和权重通常是只读的，输出是写入的
        # 检查输出地址是否与其他地址重叠
        out_size = 1  # 需要从其他信息获取实际大小

        # 从R/C/M计算输出大小
        try:
            R = parse_addr(layer.get('R', '1'))
            C = parse_addr(layer.get('C', '1'))
            M = parse_addr(layer.get('M', '1'))
            out_size = R * C * M
        except:
            pass

        addr_ranges.append({
            'index': i,
            'type': op_type,
            'in_addr': in_addr,
            'weight_addr': weight_addr,
            'out_addr': out_addr,
            'out_size': out_size,
            'out_end': out_addr + out_size
        })

    # 检查输出地址是否有重叠
    conflicts = []
    for i in range(len(addr_ranges)):
        for j in range(i + 1, len(addr_ranges)):
            r1 = addr_ranges[i]
            r2 = addr_ranges[j]

            # 检查输出地址重叠
            if r1['out_addr'] < r2['out_end'] and r2['out_addr'] < r1['out_end']:
                conflicts.append({
                    'layer1': r1['index'],
                    'layer2': r2['index'],
                    'type': 'output_overlap',
                    'addr1': r1['out_addr'],
                    'addr2': r2['out_addr']
                })

    if conflicts:
        print(f"发现 {len(conflicts)} 个地址冲突:")
        for c in conflicts[:10]:  # 只显示前10个
            print(f"  Layer {c['layer1']} 和 Layer {c['layer2']} 输出地址重叠: 0x{c['addr1']:x} vs 0x{c['addr2']:x}")
    else:
        print("✓ 没有地址冲突")

    return conflicts


def check_sequence_order(layers):
    """检查是否按照 MLIR 顺序执行"""
    print("\n" + "=" * 60)
    print("2. 检查执行顺序")
    print("=" * 60)

    # 在正确的顺序下，后续算子的输入应该是前面算子的输出
    # 或者是在DDR中的固定地址（如权重、bias）

    issues = []
    for i in range(len(layers) - 1):
        current = layers[i]
        next_layer = layers[i + 1]

        cur_out = parse_addr(current.get('Compute_Result_dest_address', '0'))
        next_in = parse_addr(next_layer.get('DDR_x1_address', '0'))

        # 检查下一个算子的输入是否是前一个算子的输出
        # （允许权重地址作为输入，因为权重是固定存储的）
        next_weight = parse_addr(next_layer.get('DDR_x2_address', '0'))

        # 如果输入不是前一个输出，也不是固定权重地址
        if next_in != cur_out and next_in != next_weight:
            # 检查输入地址是否在DDR范围内（0x10000000+是DDR地址）
            if next_in >= 0x10000000:
                issues.append({
                    'layer': i + 1,
                    'expected_in': cur_out,
                    'actual_in': next_in
                })

    if issues:
        print(f"发现 {len(issues)} 个可能的顺序问题:")
        for issue in issues[:10]:
            print(f"  Layer {issue['layer']}: 输入=0x{issue['actual_in']:x}, 期望=0x{issue['expected_in']:x}")
    else:
        print("✓ 执行顺序正确")

    return issues


def check_data_dependencies(layers):
    """检查数据依赖：输出在被使用前不应被覆盖"""
    print("\n" + "=" * 60)
    print("3. 检查数据依赖（输出在被使用前是否被覆盖）")
    print("=" * 60)

    # 为每个地址维护一个"最后使用时间"
    # 如果某个输出在后面被使用，但中间被覆盖了，就是问题

    # 构建地址使用记录
    addr_usage = {}  # address -> (layer_index, is_input)

    issues = []

    for i, layer in enumerate(layers):
        in_addr = parse_addr(layer.get('DDR_x1_address', '0'))
        in2_addr = parse_addr(layer.get('DDR_x2_address', '0'))  # 对于add算子
        out_addr = parse_addr(layer.get('Compute_Result_dest_address', '0'))

        # 检查输入地址是否被覆盖
        for used_addr, (last_layer, is_output) in addr_usage.items():
            if is_output:  # 这是一个输出地址
                # 检查是否在输入范围内
                in_size = parse_addr(layer.get('in_size', '1'))
                if used_addr == in_addr or used_addr == in2_addr:
                    # 输入使用了之前的输出
                    # 检查输出是否在其他输入之前被覆盖
                    pass

        # 记录当前层的输出地址
        out_size = 1
        try:
            R = parse_addr(layer.get('R', '1'))
            C = parse_addr(layer.get('C', '1'))
            M = parse_addr(layer.get('M', '1'))
            out_size = R * C * M
        except:
            pass

        # 检查当前输出是否与后续的输入冲突
        for j in range(i + 1, len(layers)):
            future = layers[j]
            future_in = parse_addr(future.get('DDR_x1_address', '0'))
            future_in2 = parse_addr(future.get('DDR_x2_address', '0'))
            future_weight = parse_addr(future.get('DDR_x2_address', '0'))

            # 如果未来需要这个地址作为输入，但现在就要覆盖它
            if out_addr == future_in or out_addr == future_in2:
                # 检查是否有中间层覆盖了这个地址
                for k in range(i + 1, j):
                    mid = layers[k]
                    mid_out = parse_addr(mid.get('Compute_Result_dest_address', '0'))
                    if mid_out == out_addr:
                        issues.append({
                            'producer': i,
                            'consumer': j,
                            'overwriter': k,
                            'address': out_addr
                        })
                        break

        # 更新地址使用记录
        addr_usage[out_addr] = (i, True)

    if issues:
        print(f"发现 {len(issues)} 个数据依赖问题:")
        for issue in issues[:10]:
            print(f"  Layer {issue['producer']} 产生数据 @ 0x{issue['address']:x}")
            print(f"    -> 被 Layer {issue['overwriter']} 覆盖")
            print(f"    -> Layer {issue['consumer']} 需要使用")
    else:
        print("✓ 没有数据依赖问题")

    return issues


def main():
    # 解析命令行参数
    if len(sys.argv) > 1:
        hex_path = sys.argv[1]
    else:
        # 默认路径
        hex_path = os.path.join(os.path.dirname(__file__), '..', 'parameters', 'vliw_hex.txt')
        hex_path = os.path.abspath(hex_path)

    print(f"验证文件: {hex_path}")
    print()

    if not os.path.exists(hex_path):
        print(f"错误: 文件不存在 {hex_path}")
        sys.exit(1)

    # 解析 VLIW hex
    layers = parse_vliw_hex(hex_path)
    print(f"共解析 {len(layers)} 个算子")

    # 执行检查
    conflicts = check_address_conflicts(layers)
    seq_issues = check_sequence_order(layers)
    dep_issues = check_data_dependencies(layers)

    # 总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_issues = len(conflicts) + len(seq_issues) + len(dep_issues)
    if total_issues == 0:
        print("[OK] All checks passed!")
    else:
        print(f"[ERROR] Found {total_issues} issues:")
        print(f"  - Address conflicts: {len(conflicts)}")
        print(f"  - Sequence issues: {len(seq_issues)}")
        print(f"  - Data dependency: {len(dep_issues)}")


if __name__ == '__main__':
    main()
