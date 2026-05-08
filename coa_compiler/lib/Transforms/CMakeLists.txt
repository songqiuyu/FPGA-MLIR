add_mlir_dialect_library(MLIRCOATransforms
    ShapeInfer.cpp
    OpFusion.cpp
    Tiling.cpp
    AddrAssign.cpp
    Legalize.cpp
    RegisterPasses.cpp

    ADDITIONAL_HEADER_DIRS
    ${COA_SOURCE_DIR}/include/COA

    DEPENDS
    MLIRCOAOpsIncGen
    MLIRCOAPassesIncGen

    LINK_LIBS PUBLIC
    MLIRCOADialect
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRSupport
)
