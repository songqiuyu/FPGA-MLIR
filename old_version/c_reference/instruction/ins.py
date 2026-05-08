#!/usr/bin/env python3
# bin2coe_512_little_endian.py
# 用法: python bin2coe_512_little_endian.py input.bin output.coe

import sys
from pathlib import Path

WORD_BITS = 512               # 每个向量 512 bit
WORD_BYTES = WORD_BITS // 8   # 64 字节

def bin_to_coe(in_path: str, out_path: str):
    in_path = Path(in_path)
    out_path = Path(out_path)

    if not in_path.is_file():
        raise FileNotFoundError(f"Input not found: {in_path}")

    with in_path.open('rb') as f_in, out_path.open('w', encoding='utf-8') as f_out:
        # 写 COE 头
        f_out.write("memory_initialization_radix = 16;\n")
        f_out.write("memory_initialization_vector =\n")

        first = True
        while True:
            chunk = f_in.read(WORD_BYTES)
            if not chunk:
                break

            # 最后一块不足 64B 补零
            if len(chunk) < WORD_BYTES:
                chunk = chunk + bytes(WORD_BYTES - len(chunk))

            # 小端：文件里前面的字节在低位 → 打印时要整体反转
            chunk = chunk[::-1]

            hex_str = ''.join(f'{b:02X}' for b in chunk)

            if first:
                f_out.write(hex_str)
                first = False
            else:
                f_out.write(",\n" + hex_str)

        f_out.write(";\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python bin2coe_512_little_endian.py <input.bin> <output.coe>")
        sys.exit(1)
    bin_to_coe(sys.argv[1], sys.argv[2])
