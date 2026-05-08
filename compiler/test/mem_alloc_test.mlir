// Mini ResNet-style test: proper SSA activation chains + residual skip connection
// Network: conv1 → pool → conv2 → conv3 (shortcut) → add → gap → gemm
//
// Shape flow:
//   %input: (1,3,4,4)  conv1(64 ch, 3x3) → (1,64,4,4)
//   pool1: (1,64,4,4)  maxpool(2x2)       → (1,64,2,2)
//   conv2: (1,64,2,2)  conv(64 ch, 1x1)   → (1,64,2,2)
//   conv_sc: pool1     conv(64 ch, 1x1)   → (1,64,2,2)   [skip]
//   add: conv2 + conv_sc                  → (1,64,2,2)
//   gap: (1,64,2,2)  global avg pool      → (1,64,1,1)
//   gemm: (1,64)  → (1,10)
//
// Expected liveness:
//   %input    live [-, -]  addr=0x0 (network input, pre-assigned)
//   %r_conv1  live [0, 1]  size=64*4*4=1024   B  (expires after pool1)
//   %r_pool1  live [1, 3]  size=64*2*2=256    B  (used by conv2 AND conv_sc)
//   %r_conv2  live [2, 4]  size=64*2*2=256    B  (can reuse %r_conv1's slot)
//   %r_csc    live [3, 4]  size=64*2*2=256    B
//   %r_add    live [4, 5]  size=64*2*2=256    B  (can reuse %r_pool1's slot)
//   %r_gap    live [5, 6]  size=64            B  (can reuse smaller freed blocks)
//   %r_gemm   live [6, 7]  size=10            B
//
module {
  func.func @mini_resnet(
      %input: tensor<1x3x4x4xi8>,
      %w1: tensor<64x3x3x3xi8>, %b1: tensor<64xi32>,
      %w2: tensor<64x64x1x1xi8>, %b2: tensor<64xi32>,
      %wsc: tensor<64x64x1x1xi8>, %bsc: tensor<64xi32>,
      %wg: tensor<10x64xi8>, %bg: tensor<10xi32>
  ) -> tensor<1x10xi8> {

    // conv1: (1,3,4,4) → (1,64,4,4)
    %r_conv1 = "coa.qlinearconv"(%input, %w1, %b1) {
      in_scale = 0.03 : f64, in_zp = 0 : i32,
      weight_scale = [0.002 : f64], weight_zp = [0 : i64],
      out_scale = 0.01 : f64, out_zp = -128 : i32,
      kernel_shape = [3 : i64, 3 : i64],
      strides = [1 : i64, 1 : i64],
      pads = [1 : i64, 1 : i64, 1 : i64, 1 : i64],
      dilations = [1 : i64, 1 : i64],
      N = 3, M = 64, R = 4, C = 4, R0 = 4, C0 = 4,
      tM = 64, tR = 4, tC = 4
    } : (tensor<1x3x4x4xi8>, tensor<64x3x3x3xi8>, tensor<64xi32>) -> tensor<?xi8>

    // pool1: (1,64,4,4) → (1,64,2,2)  — shortcut branch also reads pool1 later
    %r_pool1 = "coa.maxpool"(%r_conv1) {
      kernel_shape = [2 : i64, 2 : i64],
      strides = [2 : i64, 2 : i64],
      pads = [0 : i64, 0 : i64, 0 : i64, 0 : i64],
      M = 64, R = 2, C = 2, R0 = 4, C0 = 4,
      tM = 64, tR = 2, tC = 2
    } : (tensor<?xi8>) -> tensor<?xi8>

    // conv2 (main path): pool1 → (1,64,2,2)
    %r_conv2 = "coa.qlinearconv"(%r_pool1, %w2, %b2) {
      in_scale = 0.01 : f64, in_zp = -128 : i32,
      weight_scale = [0.003 : f64], weight_zp = [0 : i64],
      out_scale = 0.008 : f64, out_zp = -128 : i32,
      kernel_shape = [1 : i64, 1 : i64],
      strides = [1 : i64, 1 : i64],
      pads = [0 : i64, 0 : i64, 0 : i64, 0 : i64],
      dilations = [1 : i64, 1 : i64],
      N = 64, M = 64, R = 2, C = 2, R0 = 2, C0 = 2,
      tM = 64, tR = 2, tC = 2
    } : (tensor<?xi8>, tensor<64x64x1x1xi8>, tensor<64xi32>) -> tensor<?xi8>

    // conv_sc (skip connection): pool1 → (1,64,2,2)
    // Note: pool1 must still be live here!
    %r_csc = "coa.qlinearconv"(%r_pool1, %wsc, %bsc) {
      in_scale = 0.01 : f64, in_zp = -128 : i32,
      weight_scale = [0.004 : f64], weight_zp = [0 : i64],
      out_scale = 0.009 : f64, out_zp = 0 : i32,
      kernel_shape = [1 : i64, 1 : i64],
      strides = [1 : i64, 1 : i64],
      pads = [0 : i64, 0 : i64, 0 : i64, 0 : i64],
      dilations = [1 : i64, 1 : i64],
      N = 64, M = 64, R = 2, C = 2, R0 = 2, C0 = 2,
      tM = 64, tR = 2, tC = 2
    } : (tensor<?xi8>, tensor<64x64x1x1xi8>, tensor<64xi32>) -> tensor<?xi8>

    // add: conv2 + conv_sc → (1,64,2,2)
    %r_add = "coa.qlinearadd"(%r_conv2, %r_csc) {
      a_scale = 0.008 : f64, a_zp = -128 : i32,
      b_scale = 0.009 : f64, b_zp = 0 : i32,
      out_scale = 0.007 : f64, out_zp = -128 : i32,
      M = 64, R = 2, C = 2
    } : (tensor<?xi8>, tensor<?xi8>) -> tensor<?xi8>

    // gap: (1,64,2,2) → (1,64,1,1)
    %r_gap = "coa.qlinearglobalaveragepool"(%r_add) {
      in_scale = 0.007 : f64, in_zp = -128 : i32,
      out_scale = 0.006 : f64, out_zp = -128 : i32,
      M = 64, R = 1, C = 1, tM = 64, tR = 1, tC = 1
    } : (tensor<?xi8>) -> tensor<?xi8>

    // gemm: (1,64) → (1,10)
    %r_gemm = "coa.qgemm"(%r_gap, %wg, %bg) {
      a_scale = 0.006 : f64, a_zp = -128 : i32,
      b_scale = [0.005 : f64], b_zp = [0 : i64],
      out_scale = 0.05 : f64, out_zp = -29 : i32,
      M = 10, N = 64, R = 1, C = 1, R0 = 1, C0 = 1,
      tM = 10, tR = 1, tC = 1
    } : (tensor<?xi8>, tensor<10x64xi8>, tensor<10xi32>) -> tensor<1x10xi8>

    return %r_gemm : tensor<1x10xi8>
  }
}
