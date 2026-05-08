add_mlir_dialect_library(MLIRCOADialect
    COADialect.cpp
    COAOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${COA_SOURCE_DIR}/include/COA

    DEPENDS
    MLIRCOAOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRSupport
)
