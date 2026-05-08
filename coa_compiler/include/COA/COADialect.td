//===- COADialect.td - COA dialect definition ---------------*- tablegen -*-===//
//
// Defines the COA (Coarse-grained Operator Accelerator) dialect for targeting
// a custom FPGA VLIW instruction set.
//
//===----------------------------------------------------------------------===//

#ifndef COA_DIALECT
#define COA_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// COA Dialect
//===----------------------------------------------------------------------===//

def COA_Dialect : Dialect {
    let name = "coa";
    let summary = "Coarse-grained Operator Accelerator MLIR dialect.";
    let description = [{
        The COA dialect models the coarse-grained quantized neural network
        operators supported by the custom FPGA VLIW accelerator. It covers
        the full compiler pipeline from ONNX-imported high-level ops to
        hardware-lowered ops carrying DDR addresses, tiling parameters, and
        quantization factors ready for 512-bit VLIW binary emission.

        Supported operators:
          coa.qlinearconv          - Quantized 2-D convolution (INT8)
          coa.qgemm                - Quantized general matrix multiply (INT8)
          coa.maxpool              - Integer max pooling
          coa.qlinearadd           - Quantized element-wise add (residual)
          coa.qlinearglobalaveragepool - Quantized global average pooling
    }];
    let cppNamespace = "::mlir::coa";
    let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base operation class
//===----------------------------------------------------------------------===//

class COA_Op<string mnemonic, list<Trait> traits = []> :
        Op<COA_Dialect, mnemonic, traits>;

#endif // COA_DIALECT
