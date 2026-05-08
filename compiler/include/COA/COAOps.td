//===- COAOps.td - COA dialect operation definitions --------*- tablegen -*-===//
//
// Defines all COA dialect operations with their ONNX-level attributes and
// hardware-assigned attributes (addresses, tiling, quantization factors).
//
//===----------------------------------------------------------------------===//

#ifndef COA_OPS
#define COA_OPS

include "COA/COADialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Shared hardware attribute groups (assigned by compiler passes)
//===----------------------------------------------------------------------===//

// DDR address attributes injected by --coa-addr-assign
def HW_InAddr    : DefaultValuedAttr<I64Attr, "0"> { let storageType = [{ ::mlir::IntegerAttr }]; let returnType = [{ int64_t }]; }

//===----------------------------------------------------------------------===//
// coa.qlinearconv  -  Quantized 2-D Convolution
//===----------------------------------------------------------------------===//

def COA_QLinearConvOp : COA_Op<"qlinearconv"> {
    let summary = "INT8 quantized 2-D convolution targeting FPGA VLIW accelerator.";
    let description = [{
        Performs a quantized linear 2-D convolution with per-channel or per-tensor
        scale/zero-point quantization, mapping to VLIW operator type 0 (Conv).

        High-level attributes (from ONNX):
          in_scale / in_zp        - Input activation quantization
          weight_scale / weight_zp - Weight quantization (per-channel: array)
          out_scale / out_zp      - Output activation quantization
          kernel_shape, strides, pads, dilations - Convolution geometry

        Hardware attributes (filled by passes, default 0):
          in_addr, out_addr, weight_addr, bias_addr, silu_addr
          R, C, M, N, R0, C0     - Tensor dimensions
          tM, tR, tC             - Tile sizes
          factor                 - Fixed-point quantization multiplier
          sM_concat, M_concat    - Concat channel metadata
    }];

    let arguments = (ins
        AnyTensor:$input,
        AnyTensor:$weight,
        Optional<AnyTensor>:$bias,

        // ONNX-level quantization parameters
        F64Attr:$in_scale,
        I32Attr:$in_zp,
        F64ArrayAttr:$weight_scale,        // per-channel array
        I64ArrayAttr:$weight_zp,            // per-channel array
        F64Attr:$out_scale,
        I32Attr:$out_zp,

        // Convolution geometry
        I64ArrayAttr:$kernel_shape,
        OptionalAttr<I64ArrayAttr>:$strides,
        OptionalAttr<I64ArrayAttr>:$pads,
        OptionalAttr<I64ArrayAttr>:$dilations,

        // Hardware-assigned by --coa-addr-assign (default 0)
        DefaultValuedAttr<I64Attr, "0">:$in_addr,
        DefaultValuedAttr<I64Attr, "0">:$out_addr,
        DefaultValuedAttr<I64Attr, "0">:$weight_addr,
        DefaultValuedAttr<I64Attr, "0">:$bias_addr,
        DefaultValuedAttr<I64Attr, "0">:$silu_addr,

        // Tensor dimensions - filled by --coa-shape-infer
        DefaultValuedAttr<I64Attr, "0">:$R,
        DefaultValuedAttr<I64Attr, "0">:$C,
        DefaultValuedAttr<I64Attr, "0">:$M,
        DefaultValuedAttr<I64Attr, "0">:$N,
        DefaultValuedAttr<I64Attr, "0">:$R0,
        DefaultValuedAttr<I64Attr, "0">:$C0,
        DefaultValuedAttr<I64Attr, "0">:$sM_concat,
        DefaultValuedAttr<I64Attr, "0">:$M_concat,

        // Tiling - filled by --coa-tiling
        DefaultValuedAttr<I64Attr, "0">:$tM,
        DefaultValuedAttr<I64Attr, "0">:$tR,
        DefaultValuedAttr<I64Attr, "0">:$tC,

        // Quantization factor - filled by --coa-addr-assign
        DefaultValuedAttr<I64Attr, "0">:$factor
    );

    let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// coa.maxpool  -  Integer Max Pooling
//===----------------------------------------------------------------------===//

def COA_MaxPoolOp : COA_Op<"maxpool"> {
    let summary = "Integer max pooling, maps to VLIW operator type 1 (Pool).";
    let description = [{
        Performs max pooling over INT8 or quantized integer input tensor.
        Maps to VLIW operator type 1.
    }];

    let arguments = (ins
        AnyTensor:$input,

        // Pooling geometry
        I64ArrayAttr:$kernel_shape,
        OptionalAttr<I64ArrayAttr>:$strides,
        OptionalAttr<I64ArrayAttr>:$pads,

        // Hardware-assigned
        DefaultValuedAttr<I64Attr, "0">:$in_addr,
        DefaultValuedAttr<I64Attr, "0">:$out_addr,

        // Dimensions
        DefaultValuedAttr<I64Attr, "0">:$R,
        DefaultValuedAttr<I64Attr, "0">:$C,
        DefaultValuedAttr<I64Attr, "0">:$M,
        DefaultValuedAttr<I64Attr, "0">:$N,
        DefaultValuedAttr<I64Attr, "0">:$R0,
        DefaultValuedAttr<I64Attr, "0">:$C0,
        DefaultValuedAttr<I64Attr, "0">:$sM_concat,
        DefaultValuedAttr<I64Attr, "0">:$M_concat,

        // Tiling
        DefaultValuedAttr<I64Attr, "0">:$tM,
        DefaultValuedAttr<I64Attr, "0">:$tR,
        DefaultValuedAttr<I64Attr, "0">:$tC
    );

    let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// coa.qlinearadd  -  Quantized Element-wise Add (Residual)
//===----------------------------------------------------------------------===//

def COA_QLinearAddOp : COA_Op<"qlinearadd"> {
    let summary = "INT8 quantized element-wise add, maps to VLIW operator type 3 (Add).";
    let description = [{
        Dequantizes two INT8 tensors using their respective scales/zero-points,
        adds them, and requantizes to the output scale/zero-point.
        Implements the residual skip-connection pattern in ResNet/YOLO.
    }];

    let arguments = (ins
        AnyTensor:$a,
        AnyTensor:$b,

        // Quantization parameters
        F64Attr:$a_scale,
        I32Attr:$a_zp,
        F64Attr:$b_scale,
        I32Attr:$b_zp,
        F64Attr:$out_scale,
        I32Attr:$out_zp,

        // Hardware-assigned
        DefaultValuedAttr<I64Attr, "0">:$in_addr,
        DefaultValuedAttr<I64Attr, "0">:$in2_addr,
        DefaultValuedAttr<I64Attr, "0">:$out_addr,
        DefaultValuedAttr<I64Attr, "0">:$factor,
        DefaultValuedAttr<I64Attr, "0">:$factor2,

        // Dimensions
        DefaultValuedAttr<I64Attr, "0">:$R,
        DefaultValuedAttr<I64Attr, "0">:$C,
        DefaultValuedAttr<I64Attr, "0">:$M,
        DefaultValuedAttr<I64Attr, "0">:$N,
        DefaultValuedAttr<I64Attr, "0">:$R0,
        DefaultValuedAttr<I64Attr, "0">:$C0,
        DefaultValuedAttr<I64Attr, "0">:$sM_concat,
        DefaultValuedAttr<I64Attr, "0">:$M_concat,

        // Tiling
        DefaultValuedAttr<I64Attr, "0">:$tM,
        DefaultValuedAttr<I64Attr, "0">:$tR,
        DefaultValuedAttr<I64Attr, "0">:$tC
    );

    let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// coa.qlinearglobalaveragepool  -  Quantized Global Average Pool
//===----------------------------------------------------------------------===//

def COA_QLinearGlobalAveragePoolOp : COA_Op<"qlinearglobalaveragepool"> {
    let summary = "INT8 quantized global average pooling, maps to VLIW operator type 2 (Upsample/GAP).";
    let description = [{
        Computes the spatial mean of an INT8 tensor and requantizes the result.
        Maps to VLIW operator type 2.
    }];

    let arguments = (ins
        AnyTensor:$input,

        // Quantization
        F64Attr:$in_scale,
        I32Attr:$in_zp,
        F64Attr:$out_scale,
        I32Attr:$out_zp,

        // Hardware-assigned
        DefaultValuedAttr<I64Attr, "0">:$in_addr,
        DefaultValuedAttr<I64Attr, "0">:$out_addr,

        // Dimensions
        DefaultValuedAttr<I64Attr, "0">:$R,
        DefaultValuedAttr<I64Attr, "0">:$C,
        DefaultValuedAttr<I64Attr, "0">:$M,
        DefaultValuedAttr<I64Attr, "0">:$N,
        DefaultValuedAttr<I64Attr, "0">:$R0,
        DefaultValuedAttr<I64Attr, "0">:$C0,

        // Tiling
        DefaultValuedAttr<I64Attr, "0">:$tM,
        DefaultValuedAttr<I64Attr, "0">:$tR,
        DefaultValuedAttr<I64Attr, "0">:$tC
    );

    let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// coa.qgemm  -  Quantized General Matrix Multiply (FC Layer)
//===----------------------------------------------------------------------===//

def COA_QGemmOp : COA_Op<"qgemm"> {
    let summary = "INT8 quantized general matrix multiply (fully-connected layer), maps to VLIW operator type 0.";
    let description = [{
        Performs INT8 GEMM: Y = A * B^T (transB=1) * alpha + C.
        Represents a fully-connected layer after flattening.
        The FPGA treats this identically to a 1x1 convolution; maps to VLIW op type 0.
    }];

    let arguments = (ins
        AnyTensor:$a,
        AnyTensor:$b,
        Optional<AnyTensor>:$c,

        // Quantization
        F64Attr:$a_scale,
        I32Attr:$a_zp,
        F64ArrayAttr:$b_scale,
        I64ArrayAttr:$b_zp,
        F64Attr:$out_scale,
        I32Attr:$out_zp,

        // GEMM parameters
        DefaultValuedAttr<I32Attr, "1">:$transB,
        DefaultValuedAttr<F32Attr, "1.0f">:$alpha,

        // Hardware-assigned
        DefaultValuedAttr<I64Attr, "0">:$in_addr,
        DefaultValuedAttr<I64Attr, "0">:$out_addr,
        DefaultValuedAttr<I64Attr, "0">:$weight_addr,
        DefaultValuedAttr<I64Attr, "0">:$bias_addr,
        DefaultValuedAttr<I64Attr, "0">:$silu_addr,

        // Dimensions
        DefaultValuedAttr<I64Attr, "0">:$R,
        DefaultValuedAttr<I64Attr, "0">:$C,
        DefaultValuedAttr<I64Attr, "0">:$M,
        DefaultValuedAttr<I64Attr, "0">:$N,
        DefaultValuedAttr<I64Attr, "0">:$R0,
        DefaultValuedAttr<I64Attr, "0">:$C0,
        DefaultValuedAttr<I64Attr, "0">:$sM_concat,
        DefaultValuedAttr<I64Attr, "0">:$M_concat,

        // Tiling
        DefaultValuedAttr<I64Attr, "0">:$tM,
        DefaultValuedAttr<I64Attr, "0">:$tR,
        DefaultValuedAttr<I64Attr, "0">:$tC,

        // Quantization factor
        DefaultValuedAttr<I64Attr, "0">:$factor
    );

    let results = (outs AnyTensor:$output);
}

#endif // COA_OPS
