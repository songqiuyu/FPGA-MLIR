//===- coa-opt.cpp - COA dialect optimizer driver -----------------*- C++ -*-===//
//
// mlir-opt style driver for the COA dialect.
// Used for developing and testing individual passes.
//
// Usage:
//   coa-opt --coa-shape-infer --coa-tiling --coa-addr-assign --coa-legalize \
//           --coa-vliw-gen=output=model.vliw model.mlir
//
//===----------------------------------------------------------------------===//

#include "COA/COADialect.h"
#include "COA/COAPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
    mlir::registerAllPasses();
    mlir::coa::registerCOAPasses();

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert<mlir::coa::COADialect>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "COA optimizer driver\n", registry));
}
