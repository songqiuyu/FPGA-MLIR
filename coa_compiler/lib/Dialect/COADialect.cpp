//===- COADialect.cpp - COA dialect implementation ----------------*- C++ -*-===//

#include "COA/COADialect.h"
#include "COA/COAOps.h"

using namespace mlir;
using namespace mlir::coa;

// Include TableGen-generated dialect definition.
#include "COA/COAOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// COA Dialect initialization
//===----------------------------------------------------------------------===//

void COADialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "COA/COAOps.cpp.inc"
    >();
}
