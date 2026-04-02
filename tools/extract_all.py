"""
整合入口: 提取所有 (weight + bias + relu + input)
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tools"))

import extract_weights
import extract_bias
import extract_relu
import extract_input


def main():
    print("=" * 60)
    print("[Extract All]")
    print("=" * 60)

    # 1. 提取权重
    print("\n[1/4] Extracting weights...")
    extract_weights.extract()

    # 2. 提取 bias
    print("\n[2/4] Extracting bias...")
    extract_bias.extract()

    # 3. 提取 relu
    print("\n[3/4] Extracting relu...")
    extract_relu.generate_relu_lut()

    # 4. 提取输入
    print("\n[4/4] Extracting input...")
    extract_input.extract()

    print("\n" + "=" * 60)
    print("[Done] All files generated in parameters/")
    print("=" * 60)


if __name__ == "__main__":
    main()