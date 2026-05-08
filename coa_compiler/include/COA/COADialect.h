//===- COADialect.h - COA dialect declaration -----------------------*- C++ -*-===//
//
// Declares the COA (Coarse-grained Operator Accelerator) MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef COA_COADIALECT_H
#define COA_COADIALECT_H

#include "mlir/IR/Dialect.h"

// Include TableGen-generated dialect declaration.
#include "COA/COAOpsDialect.cpp.inc"

#endif // COA_COADIALECT_H
