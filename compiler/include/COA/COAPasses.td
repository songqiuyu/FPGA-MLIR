//===- COAPasses.td - COA dialect pass definitions ----------*- tablegen -*-===//
//
// Declares all passes in the COA compiler pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef COA_PASSES
#define COA_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Pass 1: Shape Inference
// Propagates output tensor shapes through the COA compute graph by examining
// op attributes (kernel, stride, pad, dilation, channel counts).
//===----------------------------------------------------------------------===//

def COAShapeInfer : Pass<"coa-shape-infer", "::mlir::func::FuncOp"> {
    let summary = "Infer and annotate tensor shapes for all COA ops.";
    let constructor = "::mlir::coa::createCOAShapeInferPass()";
    let description = [{
        Walks the function body and fills in R, C, M, N, R0, C0 dimension
        attributes on each COA op based on the input tensor shapes and
        convolution geometry parameters (kernel, stride, pad, dilation).
        Must run before --coa-tiling and --coa-addr-assign.
    }];
    let dependentDialects = ["::mlir::coa::COADialect", "::mlir::func::FuncDialect"];
}

//===----------------------------------------------------------------------===//
// Pass 2: Operator Fusion
// Fuses common patterns: Conv+ReLU6, Conv+SiLU, Conv+Add (residual).
// This is the rule-based baseline; AI-guided fusion is in ai_optimizer/.
//===----------------------------------------------------------------------===//

def COAOpFusion : Pass<"coa-op-fusion", "::mlir::func::FuncOp"> {
    let summary = "Fuse Conv+Activation and Conv+Add patterns in the COA graph.";
    let constructor = "::mlir::coa::createCOAOpFusionPass()";
    let description = [{
        Pattern rewrites:
          - coa.qlinearconv -> coa.relu6  =>  fused (silu_addr / activation LUT)
          - coa.qlinearconv -> coa.qlinearadd (residual) => merged
        Sets the silu_addr attribute when an activation LUT is required.
        Must run after --coa-shape-infer.
    }];
    let dependentDialects = ["::mlir::coa::COADialect", "::mlir::func::FuncDialect"];
}

//===----------------------------------------------------------------------===//
// Pass 3: Tiling
// Computes tM / tR / tC for each convolution op respecting FPGA Buffer limits.
// Implements the get_tile() heuristic; RL replacement is in ai_optimizer/.
//===----------------------------------------------------------------------===//

def COATiling : Pass<"coa-tiling", "::mlir::func::FuncOp"> {
    let summary = "Compute hardware-aware tM/tR/tC tile sizes for each COA op.";
    let constructor = "::mlir::coa::createCOATilingPass()";
    let description = [{
        For each coa.qlinearconv / coa.qgemm:
          1. Reads N, M, R, C, kernel, stride, pad, dilation from op attributes.
          2. Runs the buffer-constraint solver (ports assign_addr.get_tile()):
               wdepth < 256, gdepth < 1024, odepth < 2048
          3. Writes tM, tR, tC back as op attributes.
        Must run after --coa-shape-infer.
    }];
    let options = [
        Option<"wdepthLimit", "wdepth-limit", "int64_t", "256",
               "Maximum weight buffer depth (in 32-element groups)">,
        Option<"gdepthLimit", "gdepth-limit", "int64_t", "1024",
               "Maximum input tile buffer depth">,
        Option<"odepthLimit", "odepth-limit", "int64_t", "2048",
               "Maximum output tile buffer depth">
    ];
    let dependentDialects = ["::mlir::coa::COADialect", "::mlir::func::FuncDialect"];
}

//===----------------------------------------------------------------------===//
// Pass 4: Address Assignment
// Assigns DDR in_addr / out_addr / weight_addr / bias_addr to each op and
// computes the fixed-point quantization factor.
//===----------------------------------------------------------------------===//

def COAAddrAssign : Pass<"coa-addr-assign", "::mlir::func::FuncOp"> {
    let summary = "Assign DDR addresses and compute quantization factors for each COA op.";
    let constructor = "::mlir::coa::createCOAAddrAssignPass()";
    let description = [{
        Address layout (matching assign_addr.py):
          - First-layer input:   0x0
          - Activation buffers:  BASE + 0x10000000 (ping-pong)
          - Weight base:         0x8000000 (IC padded to 32)
          - Bias base:           0xC0000000
        Also computes:
          factor = round(in_scale * weight_scale / out_scale * 2^15)
          factor2 = round(a_scale / out_scale * 2^15)  (for qlinearadd)
    }];
    let options = [
        Option<"weightBase", "weight-base", "int64_t", "0x8000000",
               "Base DDR address for weight tensors">,
        Option<"biasBase", "bias-base", "int64_t", "0xC0000000",
               "Base DDR address for bias tensors">,
        Option<"activationBase", "act-base", "int64_t", "0x10000000",
               "Base DDR address offset for activation buffers">
    ];
    let dependentDialects = ["::mlir::coa::COADialect", "::mlir::func::FuncDialect"];
}

//===----------------------------------------------------------------------===//
// Pass 5: Legalization / Verification
// Validates all hardware attributes are legal before VLIW emission.
//===----------------------------------------------------------------------===//

def COALegalize : Pass<"coa-legalize", "::mlir::func::FuncOp"> {
    let summary = "Verify all COA hardware attributes satisfy FPGA constraints.";
    let constructor = "::mlir::coa::createCOALegalizePass()";
    let description = [{
        Checks:
          - All addresses are non-zero (except the first layer input).
          - tM >= 16, tR >= 1, tC >= 1.
          - wdepth / gdepth / odepth are within buffer limits.
          - Dimension values R, C, M, N fit in their VLIW bit-fields
            (R/C/R0/C0 <= 2047, M/N <= 4095).
          - factor is representable as 34-bit signed integer.
        Emits an error and fails the pass if any constraint is violated.
    }];
    let dependentDialects = ["::mlir::coa::COADialect", "::mlir::func::FuncDialect"];
}

//===----------------------------------------------------------------------===//
// Pass 6: VLIW Code Generation
// Translates hardware-lowered COA MLIR to 64-byte VLIW binary.
//===----------------------------------------------------------------------===//

def COAVLIWGen : Pass<"coa-vliw-gen", "::mlir::func::FuncOp"> {
    let summary = "Emit 512-bit VLIW binary instructions from hardware-lowered COA ops.";
    let constructor = "::mlir::coa::createCOAVLIWGenPass()";
    let description = [{
        For each COA op in program order:
          1. Maps op type to VLIW operator field (0=conv, 1=pool, 2=GAP, 3=add).
          2. Packs all hardware attributes into the 512-bit VLIW bit-field layout.
          3. Appends the 64-byte instruction to the output file.
        Output is a raw binary file (N × 64 bytes) ready for FPGA DMA transfer.
    }];
    let options = [
        Option<"outputFile", "output", "std::string", "\"output.vliw\"",
               "Path to the output VLIW binary file">
    ];
    let dependentDialects = ["::mlir::coa::COADialect", "::mlir::func::FuncDialect"];
}

#endif // COA_PASSES
