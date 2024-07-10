在实现上一个任务的向量化版本时，你是如何确定向量化长度的（即，一次取多少个元素进行存取和计算操作）？请带着这个问题，参考`buddy-compiler`的动态向量支持文档和向量加法实现，实现一个与`gcc-loops`程序具有同样功能的动态类型向量化版本，文件命名为`MLIRGccLoopsEx1DynVec.mlir`，然后面向`RISC-V`平台下降得到汇编指令，并与上一个任务的向量化版本的相比较。



这个问题相较于问题2的难度，可以说是学了攻球去打马龙。这一个问题花的时间也是最久，难点在于(1)RISC-V工具链的构建。（2）测试用例该怎么弄。(3) 怎么将源代码下降到RISC-V架构，怎么用QEMU来模拟RISC-V架构运行可执行文件。

先来看MLIRGccLoopsEx1Vec.mlir，内容片段：

```
func.func @mlir_gccloopsex1vec(%A: memref<?xi32>, %B: memref<?xi32>,
                      %C: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %vector_size = arith.constant 4 : index  // 假设向量大小为4
  %n = memref.dim %B, %c0 : memref<?xi32>
```

可以看出我们设置的向量化长度为4。

动态向量加法的[实现在这](https://github.com/buddy-compiler/buddy-mlir/blob/main/examples/VectorExpDialect/vector-exp-dynamic-vector.mlir)，我们把它的函数借鉴一下就是”MLIRGccLoopsEx1DynVec.mlir“：

```
func.func @vector_add(%input1: memref<?xi32>, %input2: memref<?xi32>, %output: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  // Get the dimension of the workload.
  %dim_size = memref.dim %input1, %c0 : memref<?xi32>
  // Perform dynamic vector addition.
  // Returns four times the physical vl for element type i32.
  %vl = vector_exp.get_vl i32, 4 : index

  scf.for %idx = %c0 to %dim_size step %vl { // Tiling
    %it_vl = affine.min #map(%idx)[%vl, %dim_size]
    vector_exp.set_vl %it_vl : index {
      %vec_input1 = vector.load %input1[%idx] : memref<?xi32>, vector<[1]xi32> // vector<?xi32>
      %vec_input2 = vector.load %input2[%idx] : memref<?xi32>, vector<[1]xi32> // vector<?xi32>
      %vec_output = arith.addi %vec_input1, %vec_input2 : vector<[1]xi32> // vector<?xi32>
      vector.store %vec_output, %output[%idx] : memref<?xi32>, vector<[1]xi32> // vector<?xi32>
      vector.yield
    }
  }
  return
}
```

完成这一步，我们要给MLIRGccLoopsEx1DynVec.mlir写benchmark了：MLIRGccLoopsEx1DynVecBenchmark.cpp，内容如下：

```
//===- MLIRGccLoopsEx1DynVecBenchmark.cpp --------------------------------------------===//

#include <benchmark/benchmark.h>
#include </home/zesang/buddy-mlir/frontend/Interfaces/buddy/Core/Container.h>
#include <iostream>

// Declare the gccloopsex1dynvec C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex1dynvec(MemRef<int, 1> *output,
                              MemRef<int, 1> *input1,
                              MemRef<int, 1> *input2);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRGccLoopsEx1DynVec_1[1] = {10};
intptr_t sizesInputArrayMLIRGccLoopsEx1DynVec_2[1] = {10};
intptr_t sizesOutputArrayMLIRGccLoopsEx1DynVec[1] = {10};
// Define the MemRef container for inputs and output.
MemRef<int, 1> inputMLIRGccLoopsEx1DynVec_1(sizesInputArrayMLIRGccLoopsEx1DynVec_1, 2);
MemRef<int, 1> inputMLIRGccLoopsEx1DynVec_2(sizesInputArrayMLIRGccLoopsEx1DynVec_2, 3);
MemRef<int, 1> outputMLIRGccLoopsEx1DynVec(sizesOutputArrayMLIRGccLoopsEx1DynVec, 0);

static void MLIR_GccLoopsEx1DynVec(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex1dynvec(&outputMLIRGccLoopsEx1DynVec, &inputMLIRGccLoopsEx1DynVec_1,
                               &inputMLIRGccLoopsEx1DynVec_2);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx1DynVec)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx1DynVec() {
  // Define the MemRef descriptor for inputs and output.
  MemRef<int, 1> input1(sizesInputArrayMLIRGccLoopsEx1DynVec_1, 2);
  MemRef<int, 1> input2(sizesInputArrayMLIRGccLoopsEx1DynVec_2, 3);
  MemRef<int, 1> output(sizesOutputArrayMLIRGccLoopsEx1DynVec, 0);
  // Run the gccloopsex1.
  _mlir_ciface_mlir_gccloopsex1dynvec(&output, &input1, &input2);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx1: MLIR GccLoopsEx1DynVec Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
```


CMakeLists.txt现在得自己写一个了，可以抄buddy-benchmark的作业：

```
cmake_minimum_required(VERSION 3.10)

project(MyBenchmark)

# 设置工具链文件
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/toolchain-riscv64.cmake)


# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# 设置编译标志
set(CMAKE_CXX_FLAGS "-no-pie")
set(CMAKE_C_FLAGS "-no-pie")

# 设置 RISC-V路径
set(RISCV_GCC /home/zesang/buddy-mlir/thirdparty/build-riscv-gnu-toolchain/bin/riscv64-unknown-linux-gnu-gcc)
set(RISCV_LIB_PATH /home/zesang/buddy-mlir/thirdparty/build-cross-mlir/lib)

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
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
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


#-------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx1 Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/mlir-gccloopsex1.o
  COMMAND /home/zesang/buddy-mlir/llvm/build/bin/mlir-opt ${CMAKE_SOURCE_DIR}/MLIRGccLoopsEx1.mlir 
            -convert-scf-to-cf
            -expand-strided-metadata
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -finalize-memref-to-llvm
            -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          /home/zesang/buddy-mlir/llvm/build/bin/mlir-translate --mlir-to-llvmir |
          /home/zesang/buddy-mlir/llvm/build/bin/llc -mtriple riscv64 -mattr=+v,+m -riscv-v-vector-bits-min=128 --filetype=obj -o ${CMAKE_BINARY_DIR}/mlir-gccloopsex1.o
)

add_library(MLIRGccLoopsEx1 STATIC mlir-gccloopsex1.o)
set_target_properties(MLIRGccLoopsEx1 PROPERTIES LINKER_LANGUAGE CXX)

#------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx1Vec Operation + Upstream Lowering Passes
#------------------------------------------------------------------------------

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/mlir-gccloopsex1vec.o
  COMMAND /home/zesang/buddy-mlir/llvm/build/bin/mlir-opt ${CMAKE_SOURCE_DIR}/MLIRGccLoopsEx1Vec.mlir
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
          /home/zesang/buddy-mlir/llvm/build/bin/mlir-translate --mlir-to-llvmir |
          /home/zesang/buddy-mlir/llvm/build/bin/llc -mtriple riscv64 -mattr=+v,+m -riscv-v-vector-bits-min=128 --filetype=obj -o ${CMAKE_BINARY_DIR}/mlir-gccloopsex1vec.o
)

add_library(MLIRGccLoopsEx1Vec STATIC mlir-gccloopsex1vec.o)
set_target_properties(MLIRGccLoopsEx1Vec PROPERTIES LINKER_LANGUAGE CXX)


# 添加自定义命令以生成 mlir-gccloopsex1dynvec.o 对象文件
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/mlir-gccloopsex1dynvec.o
  COMMAND /home/zesang/buddy-mlir/build/bin/buddy-opt ${CMAKE_SOURCE_DIR}/MLIRGccLoopsEx1DynVec.mlir
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
          /home/zesang/buddy-mlir/build/bin/buddy-translate --buddy-to-llvmir |
          /home/zesang/buddy-mlir/llvm/build/bin/llc -mtriple riscv64 -mattr=+v,+m -riscv-v-vector-bits-min=128 --filetype=obj -o ${CMAKE_BINARY_DIR}/mlir-gccloopsex1dynvec.o
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
```

针对RISC v我单独写了一个文件来指定编译工具链toolchain-riscv64.cmake：

```
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER /home/zesang/buddy-mlir/thirdparty/build-riscv-gnu-toolchain/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /home/zesang/buddy-mlir/thirdparty/build-riscv-gnu-toolchain/bin/riscv64-unknown-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH /home/zesang/buddy-mlir/thirdparty/build-riscv-gnu-toolchain/sysroot)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

编译我们的benchmark (Vectorbenchmark)：

```
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv64.cmake
$ ninja Vectorbenchmark
```

最后就是用QEMU来模拟执行得到的Vectorbenchmark：

```
LD_LIBRARY_PATH=/home/zesang/buddy-mlir/thirdparty/build-cross-mlir/lib \
 /home/zesang/buddy-mlir/thirdparty/qemu/build/riscv64-linux-user/qemu-riscv64 \
 -L /home/zesang/buddy-mlir/thirdparty/build-riscv-gnu-toolchain/sysroot \
 -cpu rv64,x-v=true,vlen=128 Vectorbenchmark
```

