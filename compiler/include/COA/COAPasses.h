//===- COAPasses.h - COA compiler pass declarations ----------------*- C++ -*-===//
//
// Declares all passes in the COA compiler pipeline and provides the
// createXxxPass() factory functions.
//
//===----------------------------------------------------------------------===//

#ifndef COA_COAPASSES_H
#define COA_COAPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace mlir::coa {

std::unique_ptr<mlir::Pass> createCOAShapeInferPass();
std::unique_ptr<mlir::Pass> createCOAOpFusionPass();
std::unique_ptr<mlir::Pass> createCOATilingPass();
std::unique_ptr<mlir::Pass> createCOAAddrAssignPass();
std::unique_ptr<mlir::Pass> createCOALegalizePass();
std::unique_ptr<mlir::Pass> createCOAVLIWGenPass();
void buildCOACompilerPipeline(mlir::OpPassManager &pm);

} // namespace mlir::coa

// GEN_PASS_DECL: forward-declares the base class templates at global scope.
#define GEN_PASS_DECL
#include "COA/COAPasses.h.inc"

// GEN_PASS_REGISTRATION: inline registration helpers inside mlir::coa.
namespace mlir::coa {
#define GEN_PASS_REGISTRATION
#include "COA/COAPasses.h.inc"
} // namespace mlir::coa

#endif // COA_COAPASSES_H
