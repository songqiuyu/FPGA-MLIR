cmake_minimum_required(VERSION 3.20.0)
project(fpga-mlir
    VERSION 0.1.0
    DESCRIPTION "FPGA AI compiler: ONNX -> COA MLIR -> 512-bit VLIW binary"
    LANGUAGES CXX C
)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

#-------------------------------------------------------------------------------
# Find MLIR/LLVM installation
# Configure with: cmake -DMLIR_DIR=<llvm_build>/lib/cmake/mlir ..
#-------------------------------------------------------------------------------
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# COA paths forwarded into the compiler subdirectory
set(COA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/compiler)
set(COA_BINARY_DIR ${PROJECT_BINARY_DIR}/compiler)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${COA_SOURCE_DIR}/include)
include_directories(${COA_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#-------------------------------------------------------------------------------
# Compiler: C++ MLIR dialect + transformation passes + driver tools
#-------------------------------------------------------------------------------
add_subdirectory(compiler)

#-------------------------------------------------------------------------------
# Tests (Python) -- run with: ctest --test-dir <build_dir> -V
# Set PYTHONPATH so optimizer package is importable without installation.
#-------------------------------------------------------------------------------
enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(FPGA_MLIR_PYTHONPATH "${PROJECT_SOURCE_DIR}:${PROJECT_SOURCE_DIR}/optimizer")

add_test(
    NAME test_vliw_codegen
    COMMAND ${Python3_EXECUTABLE} -m pytest
            ${PROJECT_SOURCE_DIR}/tests/test_vliw_codegen.py -v
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
set_tests_properties(test_vliw_codegen PROPERTIES
    ENVIRONMENT "PYTHONPATH=${FPGA_MLIR_PYTHONPATH}"
    LABELS "python;vliw"
)

add_test(
    NAME test_ai_optimizer
    COMMAND ${Python3_EXECUTABLE} -m pytest
            ${PROJECT_SOURCE_DIR}/tests/test_ai_optimizer.py -v
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
set_tests_properties(test_ai_optimizer PROPERTIES
    ENVIRONMENT "PYTHONPATH=${FPGA_MLIR_PYTHONPATH}"
    LABELS "python;optimizer"
)
