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

// Pull in all TableGen-generated pass registrations.
#define GEN_PASS_REGISTRATION
#include "COA/COAPasses.h.inc"

void registerCOAPasses() {
    // The GEN_PASS_REGISTRATION macro above expands to individual
    // register<PassName>() calls.  We call the generated aggregate here.
    registerPasses();
}

void buildCOACompilerPipeline(mlir::OpPassManager &pm) {
    pm.addPass(createCOAShapeInferPass());
    pm.addPass(createCOAOpFusionPass());
    pm.addPass(createCOATilingPass());
    pm.addPass(createCOAAddrAssignPass());
    pm.addPass(createCOALegalizePass());
    pm.addPass(createCOAVLIWGenPass());
}

} // namespace mlir::coa
