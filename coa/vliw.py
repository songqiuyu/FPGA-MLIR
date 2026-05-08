"""
coa.vliw — 512-bit (64-byte) VLIW instruction for the FPGA coarse-grained
accelerator.

Python port of:
  compiler/include/COA/VLIWDefs.h   (field layout)
  compiler/lib/CodeGen/VLIWCodeGen.cpp  (toBytes() bit-packing)

Bit-field layout (LSB-first within each byte, matches FPGA DMA expectation):
  [  7: 0]  operator            8 bits   0=Conv/GEMM, 1=Pool, 2=GAP, 3=Add
  [ 43: 8]  DDR_x1_address     36 bits   input activation DDR address
  [ 79:44]  DDR_x2_address     36 bits   weight DDR address
  [ 90:80]  Bias_source_address 11 bits  bias DDR address (lower 11 bits)
  [126:91]  Compute_Result_dest_address 36 bits  output DDR address
  [134:127] Activate_LUT_address  8 bits activation LUT page
  [145:135] R                  11 bits   output height
  [156:146] C                  11 bits   output width
  [168:157] M                  12 bits   output channels
  [180:169] N                  12 bits   input channels
  [191:181] R0                 11 bits   input height
  [202:192] C0                 11 bits   input width
  [214:203] sM_concat          12 bits
  [226:215] M_concat           12 bits
  [234:227] Quant_x1_z          8 bits   input zero-point
  [242:235] Quant_x2_z          8 bits   weight zero-point
  [250:243] Quant_y_z           8 bits   output zero-point
  [253:251] Conv_pad             3 bits
  [258:254] Conv_kernel          5 bits
  [261:259] Conv_stride          3 bits
  [264:262] Conv_dilation        3 bits
  [275:265] Conv_tR             11 bits
  [286:276] Conv_tC             11 bits
  [298:287] Conv_tM             12 bits
  [310:299] Conv_tN             12 bits
  [312:311] Conv_permuteR        2 bits
  [314:313] Conv_permuteC        2 bits
  [316:315] Conv_permuteM        2 bits
  [318:317] Conv_permuteN        2 bits
  [352:319] Conv_quant_factor   34 bits  signed
  [384:353] Conv_quant_factor2  32 bits  signed
  [511:385] (reserved, 127 bits, always 0)
"""


class VLIW:
    """
    512-bit VLIW instruction.

    Constructor accepts keyword arguments matching the field names above.
    Unknown keyword arguments are silently ignored so that layer dicts from
    mlir_parser.parse_all_layers() can be passed directly as **kwargs.
    """

    OP_CONV = 0
    OP_POOL = 1
    OP_GAP  = 2
    OP_ADD  = 3

    def __init__(self, *,
                 operator: int = 0,
                 DDR_x1_address: int = 0,
                 DDR_x2_address: int = 0,
                 Bias_source_address: int = 0,
                 Compute_Result_dest_address: int = 0,
                 Activate_LUT_address: int = 0,
                 R: int = 0,
                 C: int = 0,
                 M: int = 0,
                 N: int = 0,
                 R0: int = 0,
                 C0: int = 0,
                 sM_concat: int = 0,
                 M_concat: int = 0,
                 Quant_x1_z: int = 0,
                 Quant_x2_z: int = 0,
                 Quant_y_z: int = 0,
                 Conv_pad: int = 0,
                 Conv_kernel: int = 0,
                 Conv_stride: int = 0,
                 Conv_dilation: int = 0,
                 Conv_tR: int = 0,
                 Conv_tC: int = 0,
                 Conv_tM: int = 0,
                 Conv_tN: int = 0,
                 Conv_permuteR: int = 0,
                 Conv_permuteC: int = 0,
                 Conv_permuteM: int = 0,
                 Conv_permuteN: int = 0,
                 Conv_quant_factor: int = 0,
                 Conv_quant_factor2: int = 0,
                 **_ignored):
        self.operator                    = operator
        self.DDR_x1_address              = DDR_x1_address
        self.DDR_x2_address              = DDR_x2_address
        self.Bias_source_address         = Bias_source_address
        self.Compute_Result_dest_address = Compute_Result_dest_address
        self.Activate_LUT_address        = Activate_LUT_address
        self.R              = R
        self.C              = C
        self.M              = M
        self.N              = N
        self.R0             = R0
        self.C0             = C0
        self.sM_concat      = sM_concat
        self.M_concat       = M_concat
        self.Quant_x1_z     = Quant_x1_z
        self.Quant_x2_z     = Quant_x2_z
        self.Quant_y_z      = Quant_y_z
        self.Conv_pad       = Conv_pad
        self.Conv_kernel    = Conv_kernel
        self.Conv_stride    = Conv_stride
        self.Conv_dilation  = Conv_dilation
        self.Conv_tR        = Conv_tR
        self.Conv_tC        = Conv_tC
        self.Conv_tM        = Conv_tM
        self.Conv_tN        = Conv_tN
        self.Conv_permuteR  = Conv_permuteR
        self.Conv_permuteC  = Conv_permuteC
        self.Conv_permuteM  = Conv_permuteM
        self.Conv_permuteN  = Conv_permuteN
        self.Conv_quant_factor  = Conv_quant_factor
        self.Conv_quant_factor2 = Conv_quant_factor2

    def to_bytes(self) -> bytes:
        """
        Serialize to 64 bytes (512 bits), LSB-first per field.
        Layout matches VLIWDefs.h and the FPGA DMA expectation.
        Equivalent to C++ VLIW::toBytes().
        """
        fields = [
            (self.operator,                              8),
            (self.DDR_x1_address,                       36),
            (self.DDR_x2_address,                       36),
            (self.Bias_source_address,                  11),
            (self.Compute_Result_dest_address,          36),
            (self.Activate_LUT_address,                  8),
            (self.R,                                    11),
            (self.C,                                    11),
            (self.M,                                    12),
            (self.N,                                    12),
            (self.R0,                                   11),
            (self.C0,                                   11),
            (self.sM_concat,                            12),
            (self.M_concat,                             12),
            (self.Quant_x1_z & 0xFF,                    8),
            (self.Quant_x2_z & 0xFF,                    8),
            (self.Quant_y_z  & 0xFF,                    8),
            (self.Conv_pad,                              3),
            (self.Conv_kernel,                           5),
            (self.Conv_stride,                           3),
            (self.Conv_dilation,                         3),
            (self.Conv_tR,                              11),
            (self.Conv_tC,                              11),
            (self.Conv_tM,                              12),
            (self.Conv_tN,                              12),
            (self.Conv_permuteR,                         2),
            (self.Conv_permuteC,                         2),
            (self.Conv_permuteM,                         2),
            (self.Conv_permuteN,                         2),
            (self.Conv_quant_factor  & 0x3_FFFF_FFFF,  34),
            (self.Conv_quant_factor2 & 0xFFFF_FFFF,    32),
            (0,                                        127),  # reserved
        ]

        out = bytearray(64)
        bit_pos = 0
        for value, n_bits in fields:
            value = int(value) & ((1 << n_bits) - 1)
            for i in range(n_bits):
                byte_idx = bit_pos >> 3
                bit_idx  = bit_pos & 7
                if byte_idx < 64:
                    out[byte_idx] |= ((value >> i) & 1) << bit_idx
                bit_pos += 1
        return bytes(out)

    def __repr__(self) -> str:
        return (f"VLIW(op={self.operator} "
                f"DDR_x1=0x{self.DDR_x1_address:x} "
                f"R={self.R} C={self.C} M={self.M} N={self.N})")
