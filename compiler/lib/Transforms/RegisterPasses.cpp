//===- RegisterPasses.cpp - COA pass registration ----------------*- C++ -*-===//
//
// Registers all COA passes with the MLIR global pass registry so that
// mlir-opt / coa-opt can discover them via --coa-* flags.
//
//===----------------------------------------------------------------------===//

#include "COA/COAPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::coa {

void buildCOACompilerPipeline(mlir::OpPassManager &pm) {
    pm.addPass(createCOAShapeInferPass());
    pm.addPass(createCOAOpFusionPass());
    pm.addPass(createCOATilingPass());
    pm.addPass(createCOAAddrAssignPass());
    pm.addPass(createCOALegalizePass());
    pm.addPass(createCOAVLIWGenPass());
}

} // namespace mlir::coa
