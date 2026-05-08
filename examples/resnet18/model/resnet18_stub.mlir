// ResNet-18 COA MLIR 参考片段（Level-1，前 3 层）
// 由 02_import_mlir.py 生成的完整文件结构样例。
//
// 属性说明：
//   Level-1（此文件已有）: ONNX 量化参数 + 算子拓扑
//   Level-2（--coa-shape-infer 填入）: R, C, M, N, R0, C0
//   Level-3（--coa-tiling / --coa-addr-assign 填入）: tM, tR, tC, *_addr, factor
//
// 运行编译器后 output/resnet18_lowered.mlir 将包含所有层级的属性。

module {
  func.func @resnet18(%input: tensor<1x3x224x224xi8>) -> tensor<1x1000xi8> {

    // ── Layer 0: Conv1  3→64, 7×7, stride=2, pad=3 ─────────────────────────
    // Input:  1×3×224×224   Output: 1×64×112×112
    %conv1 = "coa.qlinearconv"(%input, %conv1_weight, %conv1_bias) {
      // Level-1: ONNX 量化参数
      in_scale  = 0.00392157 : f32,   in_zp  = -128 : i32,
      weight_scale = [0.003500 : f32],
      weight_zp    = [0 : i32],
      out_scale = 0.00980392 : f32,   out_zp = -128 : i32,
      // Level-1: 卷积几何
      kernel_shape = [7, 7],
      strides      = [2, 2],
      pads         = [3, 3, 3, 3],
      dilations    = [1, 1]
      // Level-2/3: 由编译器填入
      // R=112, C=112, M=64, N=3, R0=224, C0=224
      // tM=64, tR=?, tC=?, in_addr=0x..., weight_addr=0x08000000, factor=?
    } : (tensor<1x3x224x224xi8>,
         tensor<64x3x7x7xi8>,
         tensor<64xi32>) -> tensor<1x64x112x112xi8>

    // ── Layer 1: MaxPool  3×3, stride=2, pad=1 ──────────────────────────────
    // Input:  1×64×112×112   Output: 1×64×56×56
    %pool1 = "coa.maxpool"(%conv1) {
      in_scale  = 0.00980392 : f32,   in_zp  = -128 : i32,
      out_scale = 0.00980392 : f32,   out_zp = -128 : i32,
      kernel_shape = [3, 3],
      strides      = [2, 2],
      pads         = [1, 1, 1, 1]
    } : (tensor<1x64x112x112xi8>) -> tensor<1x64x56x56xi8>

    // ── Layer 2: ResBlock1a - Conv2a  64→64, 3×3, stride=1, pad=1 ───────────
    // Input:  1×64×56×56    Output: 1×64×56×56
    %conv2a = "coa.qlinearconv"(%pool1, %conv2a_weight, %conv2a_bias) {
      in_scale  = 0.00980392 : f32,   in_zp  = -128 : i32,
      weight_scale = [0.002800 : f32],
      weight_zp    = [0 : i32],
      out_scale = 0.00784314 : f32,   out_zp = -128 : i32,
      kernel_shape = [3, 3],
      strides      = [1, 1],
      pads         = [1, 1, 1, 1],
      dilations    = [1, 1]
    } : (tensor<1x64x56x56xi8>,
         tensor<64x64x3x3xi8>,
         tensor<64xi32>) -> tensor<1x64x56x56xi8>

    // ── Layer 3: ResBlock1a - Conv2b  64→64, 3×3, stride=1, pad=1 ───────────
    %conv2b = "coa.qlinearconv"(%conv2a, %conv2b_weight, %conv2b_bias) {
      in_scale  = 0.00784314 : f32,   in_zp  = -128 : i32,
      weight_scale = [0.003100 : f32],
      weight_zp    = [0 : i32],
      out_scale = 0.00588235 : f32,   out_zp = -128 : i32,
      kernel_shape = [3, 3],
      strides      = [1, 1],
      pads         = [1, 1, 1, 1],
      dilations    = [1, 1]
    } : (tensor<1x64x56x56xi8>,
         tensor<64x64x3x3xi8>,
         tensor<64xi32>) -> tensor<1x64x56x56xi8>

    // ── Layer 4: ResBlock1a - Add (残差) ─────────────────────────────────────
    // pool1 (shortcut) + conv2b (residual)
    %add1 = "coa.qlinearadd"(%pool1, %conv2b) {
      a_scale   = 0.00980392 : f32,   a_zp   = -128 : i32,
      b_scale   = 0.00588235 : f32,   b_zp   = -128 : i32,
      out_scale = 0.01176471 : f32,   out_zp = -128 : i32
    } : (tensor<1x64x56x56xi8>,
         tensor<1x64x56x56xi8>) -> tensor<1x64x56x56xi8>

    // ── ... (layers 5~19 由 02_import_mlir.py 自动生成) ────────────────────

    // ── Layer 20: GlobalAveragePool ──────────────────────────────────────────
    // Input: 1×512×7×7   Output: 1×512×1×1
    %gap = "coa.qlinearglobalaveragepool"(%add1) {
      in_scale  = 0.01176471 : f32,   in_zp  = -128 : i32,
      out_scale = 0.01176471 : f32,   out_zp = -128 : i32
    } : (tensor<1x64x56x56xi8>) -> tensor<1x64x1x1xi8>

    // ── Layer 21: FC (GEMM)  512→1000 ───────────────────────────────────────
    %fc = "coa.qgemm"(%gap, %fc_weight, %fc_bias) {
      in_scale  = 0.01176471 : f32,   in_zp  = -128 : i32,
      weight_scale = [0.001200 : f32],
      weight_zp    = [0 : i32],
      out_scale = 0.07843137 : f32,   out_zp = -128 : i32
    } : (tensor<1x64x1x1xi8>,
         tensor<1000x64x1x1xi8>,
         tensor<1000xi32>) -> tensor<1x1000xi8>

    return %fc : tensor<1x1000xi8>
  }
}
