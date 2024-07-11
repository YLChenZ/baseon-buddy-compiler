cmake_minimum_required(VERSION 3.10)

project(MyBenchmark)


# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# 设置编译标志
set(CMAKE_CXX_FLAGS "-no-pie")
set(CMAKE_C_FLAGS "-no-pie")

#-------------------------------------------------------------------------------
#configuration
#-------------------------------------------------------------------------------
set(BuddyMLIR_DIR ${BUDDY_MLIR_BUILD_DIR}/cmake)
find_package(BuddyMLIR REQUIRED CONFIG)

# BUDDY binary directory.
set(BUDDY_MLIR_BINARY_DIR ${BUDDY_MLIR_BUILD_DIR}/bin)

# MLIR binary directory.
set(LLVM_MLIR_BINARY_DIR ${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin)
set(LLVM_MLIR_LIBRARY_DIR ${BUDDY_MLIR_BUILD_DIR}/../llvm/build/lib)

# RISC-V Lib
set(RISCV_LIB_PATH ${BUDDY_MLIR_BUILD_DIR}/../llvm/build-cross-mlir-rv/lib)

# Add BUDDY files to the include path
include_directories(${BUDDY_MAIN_INCLUDE_DIR})
include_directories(${BUDDY_MLIR_INTERFACE_DIR})
include_directories(${BUDDY_THIRDPARTY_INCLUDE_DIR})

# Helper functions.
#include(${CMAKE_SOURCE_DIR}/toolchain-riscv64.cmake)
set(RISCV_CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/toolchain-riscv64.cmake)

#-------------------------------------------------------------------------------
# Deploy google/benchmark
#-------------------------------------------------------------------------------

message(STATUS "Configuring benchmarks: google")

include(ExternalProject)

ExternalProject_Add(project_googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG "v1.6.0"
  GIT_SHALLOW 1
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/vendor/benchmark
  TIMEOUT 10
  BUILD_BYPRODUCTS <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX}
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/vendor/benchmark
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_TOOLCHAIN_FILE=${RISCV_CMAKE_TOOLCHAIN_FILE}
    -DBENCHMARK_ENABLE_TESTING=OFF
  UPDATE_COMMAND ""
  TEST_COMMAND "")

ExternalProject_Get_Property(project_googlebenchmark INSTALL_DIR)

file(MAKE_DIRECTORY ${INSTALL_DIR}/include)
add_library(GoogleBenchmark STATIC IMPORTED)
target_include_directories(GoogleBenchmark INTERFACE ${INSTALL_DIR}/include)
set_property(TARGET GoogleBenchmark PROPERTY IMPORTED_LOCATION
  "${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX}")

add_dependencies(GoogleBenchmark project_googlebenchmark)

find_package(Threads)
target_link_libraries(GoogleBenchmark INTERFACE Threads::Threads)


#-----------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx1 Operation + Upstream Lowering Passes
#-----------------------------------------------------------------------------------

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/mlir-gccloopsex1.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_SOURCE_DIR}/MLIRGccLoopsEx1.mlir 
            -convert-scf-to-cf
            -expand-strided-metadata
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -finalize-memref-to-llvm
            -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple riscv64 -mattr=+v,+m -riscv-v-vector-bits-min=128 --filetype=obj -o ${CMAKE_BINARY_DIR}/mlir-gccloopsex1.o
)

add_library(MLIRGccLoopsEx1 STATIC mlir-gccloopsex1.o)
set_target_properties(MLIRGccLoopsEx1 PROPERTIES LINKER_LANGUAGE CXX)

#---------------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx1Vec Operation + Upstream Lowering Passes
#---------------------------------------------------------------------------------------

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/mlir-gccloopsex1vec.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_SOURCE_DIR}/MLIRGccLoopsEx1Vec.mlir
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-cf
            -expand-strided-metadata
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple riscv64 -mattr=+v,+m -riscv-v-vector-bits-min=128 --filetype=obj -o ${CMAKE_BINARY_DIR}/mlir-gccloopsex1vec.o
)

add_library(MLIRGccLoopsEx1Vec STATIC mlir-gccloopsex1vec.o)
set_target_properties(MLIRGccLoopsEx1Vec PROPERTIES LINKER_LANGUAGE CXX)


#-------------------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx1DynVec Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------------------

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/mlir-gccloopsex1dynvec.o
  COMMAND ${BUDDY_MLIR_BINARY_DIR}/buddy-opt ${CMAKE_SOURCE_DIR}/MLIRGccLoopsEx1DynVec.mlir
            -lower-vector-exp
            -convert-vector-to-llvm
            -lower-affine
            -convert-scf-to-cf
            -convert-math-to-llvm
            -lower-rvv
            -convert-vector-to-llvm
            -llvm-request-c-wrappers
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple riscv64 -mattr=+v,+m -riscv-v-vector-bits-min=128 --filetype=obj -o ${CMAKE_BINARY_DIR}/mlir-gccloopsex1dynvec.o
)

add_library(MLIRGccLoopsEx1DynVec STATIC mlir-gccloopsex1dynvec.o)
set_target_properties(MLIRGccLoopsEx1DynVec PROPERTIES LINKER_LANGUAGE CXX)

# 添加源文件
add_executable(Vectorbenchmark
  Main.cpp
  MLIRGccLoopsEx1Benchmark.cpp
  MLIRGccLoopsEx1VecBenchmark.cpp
  MLIRGccLoopsEx1DynVecBenchmark.cpp)

# 链接 Google Benchmark 库和自定义 MLIR 库
target_link_libraries(Vectorbenchmark 
  GoogleBenchmark
  MLIRGccLoopsEx1
  MLIRGccLoopsEx1Vec
  MLIRGccLoopsEx1DynVec
  -L${RISCV_LIB_PATH}
  -lmlir_runner_utils
  -lmlir_c_runner_utils
  )
