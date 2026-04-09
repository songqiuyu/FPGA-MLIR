"""
VLIW 数据结构
用于 FPGA 加速器的卷积配置参数
"""

import struct
from dataclasses import dataclass
from typing import Optional


@dataclass
class VLIW:
    """VLIW 结构体 - 卷积配置参数"""

    # 操作符 (8 bits)
    operator: int = 0

    # DDR 地址 (36 bits each)
    DDR_x1_address: int = 0
    DDR_x2_address: int = 0
    Bias_source_address: int = 0      # 11 bits (实际需要扩展)
    Compute_Result_dest_address: int = 0
    Activate_LUT_address: int = 0   # 8 bits

    # 维度 (11-12 bits)
    R: int = 0       # 11 bits - 输出高度
    C: int = 0       # 11 bits - 输出宽度
    M: int = 0       # 12 bits - 输出通道
    N: int = 0       # 12 bits - 输入通道
    R0: int = 0      # 11 bits - 输入高度
    C0: int = 0      # 11 bits - 输入宽度
    sM_concat: int = 0    # 12 bits - 累加输出通道
    M_concat: int = 0     # 12 bits - concat 输出通道

    # 量化参数 (8 bits each)
    Quant_x1_z: int = 0     # 输入1 零点
    Quant_x2_z: int = 0     # 输入2 零点 (权重)
    Quant_y_z: int = 0      # 输出零点

    # 卷积参数
    Conv_pad: int = 0        # 3 bits
    Conv_kernel: int = 0      # 5 bits
    Conv_stride: int = 0      # 3 bits
    Conv_dilation: int = 0    # 3 bits

    # Tile 参数
    Conv_tR: int = 0        # 11 bits
    Conv_tC: int = 0        # 11 bits
    Conv_tM: int = 0         # 12 bits
    Conv_tN: int = 0         # 12 bits

    # 排列方式 (2 bits each)
    Conv_permuteR: int = 0
    Conv_permuteC: int = 0
    Conv_permuteM: int = 0
    Conv_permuteN: int = 0

    # 量化因子 (34 + 32 bits)
    Conv_quant_factor: int = 0     # 34 bits
    Conv_quant_factor2: int = 0    # 32 bits

    def to_bytes(self) -> bytes:
        """转换为 64 字节二进制"""
        # 按照 C 代码的顺序输出
        fields = [
            (self.operator, 8),
            (self.DDR_x1_address, 36),
            (self.DDR_x2_address, 36),
            (self.Bias_source_address, 11),
            (self.Compute_Result_dest_address, 36),
            (self.Activate_LUT_address, 8),
            (self.R, 11),
            (self.C, 11),
            (self.M, 12),
            (self.N, 12),
            (self.R0, 11),
            (self.C0, 11),
            (self.sM_concat, 12),
            (self.M_concat, 12),
            (self.Quant_x1_z, 8),
            (self.Quant_x2_z, 8),
            (self.Quant_y_z, 8),
            (self.Conv_pad, 3),
            (self.Conv_kernel, 5),
            (self.Conv_stride, 3),
            (self.Conv_dilation, 3),
            (self.Conv_tR, 11),
            (self.Conv_tC, 11),
            (self.Conv_tM, 12),
            (self.Conv_tN, 12),
            (self.Conv_permuteR, 2),
            (self.Conv_permuteC, 2),
            (self.Conv_permuteM, 2),
            (self.Conv_permuteN, 2),
            (self.Conv_quant_factor, 34),
            (self.Conv_quant_factor2, 32),
            (0, 127),  # high 131 bits not used (placeholder)
        ]

        # 构建比特流
        bits = []
        for value, num_bits in fields:
            # 按小端序存储 (最低有效位在前)
            for i in range(num_bits):
                bits.append((value >> i) & 1)

        # 转换为字节 (64 bytes = 512 bits)
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

        # Pad 到 64 字节
        while len(out_bytes) < 64:
            out_bytes.append(0)

        return bytes(out_bytes[:64])

    def __repr__(self):
        return (f"VLIW(operator={self.operator}, "
                f"DDR_x1={self.DDR_x1_address}, "
                f"DDR_x2={self.DDR_x2_address}, "
                f"R={self.R}, C={self.C}, M={self.M}, N={self.N})")


def create_vliw_example() -> VLIW:
    """创建示例 VLIW"""
    return VLIW(
        operator=1,          # Conv operator
        DDR_x1_address=0,
        DDR_x2_address=1000,
        Bias_source_address=2000,
        Compute_Result_dest_address=3000,
        Activate_LUT_address=0,
        R=224, C=224,
        M=64, N=3,
        R0=224, C0=224,
        sM_concat=64, M_concat=64,
        Quant_x1_z=0, Quant_x2_z=0, Quant_y_z=-128,
        Conv_pad=3, Conv_kernel=7, Conv_stride=2, Conv_dilation=1,
        Conv_tR=224, Conv_tC=224, Conv_tM=64, Conv_tN=3,
        Conv_permuteR=0, Conv_permuteC=0, Conv_permuteM=0, Conv_permuteN=0,
        Conv_quant_factor=0, Conv_quant_factor2=0
    )


if __name__ == "__main__":
    vliw = create_vliw_example()
    print(f"VLIW: {vliw}")
    print(f"Binary length: {len(vliw.to_bytes())} bytes")