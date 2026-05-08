//===- COADialect.h - COA dialect declaration -----------------------*- C++ -*-===//
//
// Declares the COA (Coarse-grained Operator Accelerator) MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef COA_COADIALECT_H
#define COA_COADIALECT_H

#include "mlir/IR/Dialect.h"

// Include TableGen-generated dialect declaration.
#define GET_DIALECT_CLASSES
#include "COA/COAOpsDialect.h.inc"

#endif // COA_COADIALECT_H
