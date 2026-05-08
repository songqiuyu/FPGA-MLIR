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

//===----------------------------------------------------------------------===//
// Pass factory declarations
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createCOAShapeInferPass();
std::unique_ptr<mlir::Pass> createCOAOpFusionPass();
std::unique_ptr<mlir::Pass> createCOATilingPass();
std::unique_ptr<mlir::Pass> createCOAAddrAssignPass();
std::unique_ptr<mlir::Pass> createCOALegalizePass();
std::unique_ptr<mlir::Pass> createCOAVLIWGenPass();

/// Registers all COA passes with the global pass registry.
void registerCOAPasses();

/// Returns the default full pipeline:
///   shape-infer -> op-fusion -> tiling -> addr-assign -> legalize -> vliw-gen
void buildCOACompilerPipeline(mlir::OpPassManager &pm);

} // namespace mlir::coa

// Include TableGen-generated pass declarations.
#define GEN_PASS_DECL
#include "COA/COAPasses.h.inc"

#endif // COA_COAPASSES_H
