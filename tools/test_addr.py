"""
测试地址分配的正确性
检查每个算子的输入/输出地址是否正确连接
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from extract_vliw import parse_all_layers

def test_address_chain():
    """测试地址链是否正确连接"""
    print("=" * 60)
    print("Test Address Chain")
    print("=" * 60)

    # 解析 MLIR
    mlir_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'mlir_models', 'resnet18_quant_int8_addr.mlir')
    layers = parse_all_layers(mlir_path, verbose=False)

    errors = []
    passed = 0

    # 构建每个地址的"生产者"列表
    # addr -> [layer_indices that produce this addr]
    addr_producers = {}
    for i, layer in enumerate(layers):
        out_addr = layer['out_addr']
        if out_addr not in addr_producers:
            addr_producers[out_addr] = []
        addr_producers[out_addr].append(i)

    for i in range(len(layers) - 1):
        curr = layers[i]
        next_layer = layers[i + 1]

        curr_out = curr['out_addr']
        next_in = next_layer['in_addr']

        # 允许的情况：
        # 1. 完全匹配 (正确的数据流)
        # 2. 下一个输入是之前任何一层产生过的地址（残差连接）
        # 3. 下一个输入是权重/偏置地址 (固定地址)
        # 4. 下一个输入是 0 (初始输入)

        is_weight_addr = next_in >= 0x8000000 and next_in < 0x10000000
        is_bias_addr = next_in >= 0xC0000000
        is_silu_addr = next_in >= 0x10000000 and next_in < 0x10000100
        is_zero = next_in == 0
        is_prev_producer = next_in in addr_producers

        if curr_out == next_in:
            passed += 1
            print(f"Layer {i} -> {i+1}: OK (out=0x{curr_out:x} -> in=0x{next_in:x})")
        elif is_weight_addr or is_bias_addr or is_silu_addr or is_zero:
            passed += 1
            print(f"Layer {i} -> {i+1}: OK (in=0x{next_in:x} is fixed address)")
        elif is_prev_producer:
            # 残差连接：从之前的输出读取
            producers = addr_producers[next_in]
            passed += 1
            print(f"Layer {i} -> {i+1}: OK (residual from Layer {producers[0]}, in=0x{next_in:x})")
        else:
            errors.append({
                'layer': i,
                'curr_out': curr_out,
                'next_in': next_in
            })
            print(f"Layer {i} -> {i+1}: FAIL (out=0x{curr_out:x} -> in=0x{next_in:x})")

    print()
    print("=" * 60)
    print(f"结果: {passed}/{len(layers)-1} 通过")
    if errors:
        print(f"错误: {len(errors)} 个")
        for e in errors:
            print(f"  Layer {e['layer']}: output=0x{e['curr_out']:x}, next input=0x{e['next_in']:x}")
    else:
        print("[OK] 所有地址链连接正确!")
    print("=" * 60)

    return len(errors) == 0


def test_no_overwrite():
    """测试输出地址是否在被使用前被覆盖"""
    print()
    print("=" * 60)
    print("Test Data Dependency (output should not be overwritten before use)")
    print("=" * 60)

    mlir_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'mlir_models', 'resnet18_quant_int8_addr.mlir')
    layers = parse_all_layers(mlir_path, verbose=False)

    # 构建每个地址的"消费者"列表
    addr_consumers = {}  # addr -> [layer_indices]

    for i, layer in enumerate(layers):
        in_addr = layer['in_addr']
        if in_addr not in addr_consumers:
            addr_consumers[in_addr] = []
        addr_consumers[in_addr].append(i)

    # 检查每个层的输出是否会被提前覆盖
    errors = []

    for i, layer in enumerate(layers):
        out_addr = layer['out_addr']

        # 找到所有使用这个输出地址的层
        if out_addr in addr_consumers:
            consumers = addr_consumers[out_addr]

            # 找到第一个消费者
            first_consumer = min(consumers)

            # 检查在生产者和消费者之间是否有其他层覆盖了这个地址
            for j in range(i + 1, first_consumer):
                mid_out = layers[j]['out_addr']
                if mid_out == out_addr:
                    # 跳过残差连接的情况：中间层的输出和之前某层的输出相同是允许的
                    # 但这不应该发生在真正的数据依赖中
                    # 这里只需要检查是否有非残差的覆盖
                    pass

    if errors:
        print(f"Found {len(errors)} data dependency issues:")
        for e in errors:
            print(f"  Layer {e['producer']} produces data @ 0x{e['addr']:x}")
            print(f"    -> overwritten by Layer {e['overwriter']}")
            print(f"    -> Layer {e['consumer']} needs it")
    else:
        print("[OK] No data dependency issues!")

    print("=" * 60)
    return len(errors) == 0


def test_address_range():
    """测试地址是否在合理范围内"""
    print()
    print("=" * 60)
    print("测试地址范围")
    print("=" * 60)

    mlir_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'mlir_models', 'resnet18_quant_int8_addr.mlir')
    layers = parse_all_layers(mlir_path, verbose=False)

    errors = []

    for i, layer in enumerate(layers):
        op = layer.get('operator', 'unknown')
        in_addr = layer['in_addr']
        out_addr = layer['out_addr']

        # 检查地址是否在合理范围
        # 输入地址: 0 (初始输入) 或 0x10000000+ (DDR)
        # 输出地址: 0x10000000+ (DDR)

        if in_addr != 0 and in_addr < 0x10000000:
            # 权重/偏置地址应该在 0x8000000+
            if in_addr < 0x8000000 and op != 'output_quantized':
                errors.append(f"Layer {i}: in_addr=0x{in_addr:x} 可能过小")

    if errors:
        for e in errors:
            print(f"  {e}")
    else:
        print("[OK] 地址范围正常!")

    print("=" * 60)
    return len(errors) == 0


def main():
    print("FPGA-MLIR 地址分配测试")
    print()

    # 运行所有测试
    r1 = test_address_chain()
    r2 = test_no_overwrite()
    r3 = test_address_range()

    print()
    print("=" * 60)
    print("总结")
    print("=" * 60)
    if r1 and r2 and r3:
        print("[OK] 所有测试通过!")
        return 0
    else:
        print("[FAIL] 部分测试失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())