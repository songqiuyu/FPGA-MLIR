get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRCOADialect
    MLIRCOATransforms
    MLIRCOACodeGen
    MLIROptLib
    MLIRPass
    MLIRIR
    MLIRFuncDialect
    MLIRSupport
)

add_llvm_executable(coa-opt coa-opt.cpp)
target_link_libraries(coa-opt PRIVATE ${LIBS})
llvm_update_compile_flags(coa-opt)
mlir_check_all_link_libraries(coa-opt)
