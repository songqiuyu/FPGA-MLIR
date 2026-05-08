get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRCOADialect
    MLIRCOATransforms
    MLIRCOACodeGen
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRIR
    MLIRFuncDialect
    MLIRSupport
)

add_llvm_executable(coa-compiler coa-compiler.cpp)
target_link_libraries(coa-compiler PRIVATE ${LIBS})
llvm_update_compile_flags(coa-compiler)
mlir_check_all_link_libraries(coa-compiler)
