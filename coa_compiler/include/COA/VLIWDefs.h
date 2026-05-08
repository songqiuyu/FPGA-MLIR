//===- VLIWDefs.h - FPGA VLIW instruction structure ----------------*- C++ -*-===//
//
// C++ port of tools/vliw.py.
// Defines the 512-bit (64-byte) VLIW instruction that controls the FPGA
// coarse-grained accelerator.
//
// Bit-field layout (LSB-first within each byte):
//   [  7: 0]  operator          (8 bits)  0=Conv, 1=Pool, 2=GAP/Upsample, 3=Add
//   [ 43: 8]  DDR_x1_address   (36 bits) Input tensor DDR address
//   [ 79:44]  DDR_x2_address   (36 bits) Weight tensor DDR address
//   [ 90:80]  Bias_source_addr (11 bits) Bias DDR address (upper bits external)
//   [126:91]  Result_dest_addr (36 bits) Output tensor DDR address
//   [134:127] Activate_LUT_addr (8 bits) Activation LUT DDR page
//   [145:135] R                (11 bits) Output height
//   [156:146] C                (11 bits) Output width
//   [168:157] M                (12 bits) Output channels
//   [180:169] N                (12 bits) Input channels
//   [191:181] R0               (11 bits) Input height
//   [202:192] C0               (11 bits) Input width
//   [214:203] sM_concat        (12 bits)
//   [226:215] M_concat         (12 bits)
//   [234:227] Quant_x1_z       ( 8 bits) Input zero-point
//   [242:235] Quant_x2_z       ( 8 bits) Weight zero-point
//   [250:243] Quant_y_z        ( 8 bits) Output zero-point
//   [253:251] Conv_pad         ( 3 bits)
//   [258:254] Conv_kernel      ( 5 bits)
//   [261:259] Conv_stride      ( 3 bits)
//   [264:262] Conv_dilation    ( 3 bits)
//   [275:265] Conv_tR          (11 bits)
//   [286:276] Conv_tC          (11 bits)
//   [298:287] Conv_tM          (12 bits)
//   [310:299] Conv_tN          (12 bits)
//   [312:311] Conv_permuteR    ( 2 bits)
//   [314:313] Conv_permuteC    ( 2 bits)
//   [316:315] Conv_permuteM    ( 2 bits)
//   [318:317] Conv_permuteN    ( 2 bits)
//   [352:319] Conv_quant_factor  (34 bits)
//   [384:353] Conv_quant_factor2 (32 bits)
//   [511:385] (reserved, 127 bits, set to 0)
//
//===----------------------------------------------------------------------===//

#ifndef COA_VLIWDEFS_H
#define COA_VLIWDEFS_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mlir::coa {

/// VLIW operator type encoding.
enum class VLIWOperator : uint8_t {
    Conv    = 0,  ///< Convolution / GEMM
    Pool    = 1,  ///< Max Pooling
    GAP     = 2,  ///< Global Average Pool / Upsample
    Add     = 3,  ///< Element-wise Add (residual)
};

/// 512-bit (64-byte) VLIW instruction for the FPGA coarse-grained accelerator.
struct VLIW {
    // ---- Operator ----
    uint8_t   op          = 0;   ///< 8 bits: VLIWOperator

    // ---- DDR Addresses (36 bits each) ----
    uint64_t  ddr_x1      = 0;   ///< Input activation address
    uint64_t  ddr_x2      = 0;   ///< Weight address
    uint32_t  bias_addr   = 0;   ///< Bias address (11 bits used)
    uint64_t  result_addr = 0;   ///< Output address
    uint8_t   lut_addr    = 0;   ///< Activation LUT page (8 bits)

    // ---- Tensor Dimensions ----
    uint16_t  R           = 0;   ///< Output height  (11 bits, max 2047)
    uint16_t  C           = 0;   ///< Output width   (11 bits)
    uint16_t  M           = 0;   ///< Output channels (12 bits, max 4095)
    uint16_t  N           = 0;   ///< Input channels  (12 bits)
    uint16_t  R0          = 0;   ///< Input height   (11 bits)
    uint16_t  C0          = 0;   ///< Input width    (11 bits)
    uint16_t  sM_concat   = 0;   ///< 12 bits
    uint16_t  M_concat    = 0;   ///< 12 bits

    // ---- Quantization Zero-points (8 bits each) ----
    int8_t    quant_x1_z  = 0;   ///< Input zero-point
    int8_t    quant_x2_z  = 0;   ///< Weight zero-point
    int8_t    quant_y_z   = 0;   ///< Output zero-point

    // ---- Convolution Parameters ----
    uint8_t   pad         = 0;   ///< 3 bits
    uint8_t   kernel      = 0;   ///< 5 bits
    uint8_t   stride      = 1;   ///< 3 bits
    uint8_t   dilation    = 1;   ///< 3 bits

    // ---- Tile Parameters ----
    uint16_t  tR          = 0;   ///< 11 bits
    uint16_t  tC          = 0;   ///< 11 bits
    uint16_t  tM          = 0;   ///< 12 bits
    uint16_t  tN          = 0;   ///< 12 bits (same as N for conv)

    // ---- Permute flags (2 bits each, reserved) ----
    uint8_t   permuteR    = 0;
    uint8_t   permuteC    = 0;
    uint8_t   permuteM    = 0;
    uint8_t   permuteN    = 0;

    // ---- Quantization Factors ----
    int64_t   quant_factor  = 0; ///< 34 bits signed
    int64_t   quant_factor2 = 0; ///< 32 bits signed

    /// Serialize to 64-byte little-endian bit-packed buffer.
    /// Layout matches the FPGA DMA expectation (LSB-first per field).
    std::array<uint8_t, 64> toBytes() const;

    std::string repr() const;
};

/// Pack a sequence of VLIWs into a flat byte vector (N * 64 bytes).
std::vector<uint8_t> packVLIWs(const std::vector<VLIW> &instrs);

} // namespace mlir::coa

#endif // COA_VLIWDEFS_H
