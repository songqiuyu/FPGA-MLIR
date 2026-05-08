# Generate COA dialect TableGen targets
add_mlir_dialect(COAOps coa)

# Generate pass declarations
set(LLVM_TARGET_DEFINITIONS COAPasses.td)
mlir_tablegen(COAPasses.h.inc --gen-pass-decls -name COA)
add_public_tablegen_target(MLIRCOAPassesIncGen)

# Generate dialect docs (optional)
add_mlir_doc(COADialect COADialect COA/ -gen-dialect-doc)
add_mlir_doc(COAOps COAOps COA/ -gen-op-doc)
