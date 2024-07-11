#-----------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx1 Operation + Upstream Lowering Passes
#-----------------------------------------------------------------------------------

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/mlir-gccloopsex1.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_SOURCE_DIR}/benchmark/MLIRGccLoopsEx1.mlir 
            -convert-scf-to-cf
            -expand-strided-metadata
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -finalize-memref-to-llvm
            -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple riscv64 -mattr=+v,+m -riscv-v-vector-bits-min=128 --filetype=obj -o ${CMAKE_BINARY_DIR}/benchmark/mlir-gccloopsex1.o
)

add_library(MLIRGccLoopsEx1 STATIC mlir-gccloopsex1.o)
set_target_properties(MLIRGccLoopsEx1 PROPERTIES LINKER_LANGUAGE CXX)

#---------------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx1Vec Operation + Upstream Lowering Passes
#---------------------------------------------------------------------------------------

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/mlir-gccloopsex1vec.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_SOURCE_DIR}/benchmark/MLIRGccLoopsEx1Vec.mlir
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
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple riscv64 -mattr=+v,+m -riscv-v-vector-bits-min=128 --filetype=obj -o ${CMAKE_BINARY_DIR}/benchmark/mlir-gccloopsex1vec.o
)

add_library(MLIRGccLoopsEx1Vec STATIC mlir-gccloopsex1vec.o)
set_target_properties(MLIRGccLoopsEx1Vec PROPERTIES LINKER_LANGUAGE CXX)


#-------------------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx1DynVec Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------------------

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/mlir-gccloopsex1dynvec.o
  COMMAND ${BUDDY_MLIR_BINARY_DIR}/buddy-opt ${CMAKE_SOURCE_DIR}/benchmark/MLIRGccLoopsEx1DynVec.mlir
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
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple riscv64 -mattr=+v,+m -riscv-v-vector-bits-min=128 --filetype=obj -o ${CMAKE_BINARY_DIR}/benchmark/mlir-gccloopsex1dynvec.o
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
