add_mlir_dialect_library(MLIRCOACodeGen
    VLIWCodeGen.cpp

    ADDITIONAL_HEADER_DIRS
    ${COA_SOURCE_DIR}/include/COA

    DEPENDS
    MLIRCOAOpsIncGen
    MLIRCOAPassesIncGen

    LINK_LIBS PUBLIC
    MLIRCOADialect
    MLIRCOATransforms
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRSupport
)
