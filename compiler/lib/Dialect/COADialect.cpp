//===- COADialect.cpp - COA dialect implementation ----------------*- C++ -*-===//

#include "COA/COADialect.h"
#include "COA/COAOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::coa;

// Include TableGen-generated dialect definition.
#define GET_DIALECT_CLASSES
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

// COA dialect has no custom attributes; these stubs satisfy the vtable.
mlir::Attribute COADialect::parseAttribute(mlir::DialectAsmParser &parser,
                                            mlir::Type type) const {
    parser.emitError(parser.getNameLoc(), "COA dialect has no custom attributes");
    return {};
}

void COADialect::printAttribute(mlir::Attribute attr,
                                 mlir::DialectAsmPrinter &os) const {}
