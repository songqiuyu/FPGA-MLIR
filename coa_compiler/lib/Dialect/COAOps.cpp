//===- COAOps.cpp - COA dialect operation implementations ----------*- C++ -*-===//

#include "COA/COAOps.h"
#include "COA/COADialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::coa;

// Include TableGen-generated op definitions.
#define GET_OP_CLASSES
#include "COA/COAOps.cpp.inc"
