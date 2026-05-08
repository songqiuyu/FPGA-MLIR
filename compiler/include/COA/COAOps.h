//===- COAOps.h - COA dialect operation declarations ---------------*- C++ -*-===//
//
// Declares all COA dialect operations (auto-generated from COAOps.td).
//
//===----------------------------------------------------------------------===//

#ifndef COA_COAOPS_H
#define COA_COAOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "COA/COADialect.h"

// Include TableGen-generated op declarations.
#define GET_OP_CLASSES
#include "COA/COAOps.h.inc"

#endif // COA_COAOPS_H
