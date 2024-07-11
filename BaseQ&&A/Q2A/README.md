查看上一个任务里`gcc-loops`的`MLIR`实现代码，以第一个程序`MLIRGccLoopsEx1.mlir`为例，可以看到它只是朴素的标量版本（用了`scf.for`，`arith.addi`每次只对一位进行操作）。请用`MLIR`的`Vector Dialect`实现一个同样功能的向量化版本，文件命名为`MLIRGccLoopsEx1Vec.mlir`，并加入到`buddy-benchmark`里，与`MLIRGccLoopsEx1.mlir`的性能进行比较。`Vector Dialect`的用法可参考`MLIR`官方文档或`buddy-compiler`中的已有实现，如`buddy-mlir`里的`MLIRVector`样例和`buddy-benchmark`里的矩阵乘法向量化实现。

这个问题相较于问题1提升了一个大档，要修改的东西比较多也比较细，要自己实现的倒是不多。

（1）根据要求，先写出MLIRGccLoopsEx1Vec.mlir：

内容如下：

```
func.func @mlir_gccloopsex1vec(%A: memref<?xi32>, %B: memref<?xi32>,
                      %C: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %vector_size = arith.constant 4 : index  // 假设向量大小为4
  %n = memref.dim %B, %c0 : memref<?xi32>
  // 向量化循环
  scf.for %i = %c0 to %n step %vector_size {
   // 加载向量
   %b_vec = vector.load %B[%i] : memref<?xi32>, vector<4xi32>
   %c_vec = vector.load %C[%i] : memref<?xi32>, vector<4xi32>
    // 元素相加
   %result_vec = arith.addi %b_vec, %c_vec : vector<4xi32>
    // 存储向量
   vector.store %result_vec, %A[%i] : memref<?xi32>, vector<4xi32>  
}
  return
}
```

（2）我们要将MLIRgccloopsVec.mlir加入到gccloops中的测试用例，需要改动CMakeLists.txt和Main.cpp，还需创建一个名为MLIRGccLoopsEx1VecBenchmark.cpp的文件根据MLIRGccLoopsEx1Benchmark.cpp作相应调整：

  1）改动CMakeLists.txt：
  
在原有的添加如下内容：（放在20多行）

```
#--------------------------------------------------------------------------------------
MLIR SCF Dialect GccLoopsEx1Vec Operation + Upstream Lowering Passes
#--------------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-gccloopsex1vec.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/gccloops/MLIRGccLoopsEx1Vec.mlir
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
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE}
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/gccloops/mlir-gccloopsex1vec.o
)
add_library(MLIRGccLoopsEx1Vec STATIC mlir-gccloopsex1vec.o)
set_target_properties(MLIRGccLoopsEx1Vec PROPERTIES LINKER_LANGUAGE CXX)
```

转到末尾（400多行）：

在add_executable里面添加：MLIRGccLoopsEx1VecBenchmark.cpp

在target_link_libraries里面添加：MLIRGccLoopsEx1Vec

  2）改动Main.cpp
  
在24行添加函数声明：void generateResultMLIRGccLoopsEx1Vec();

在53行添加调用函数：generateResultMLIRGccLoopsEx1Vec();

  3）写MLIRGccLoopsEx1VecBenchmark.cpp：

内容如下：

```
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>

// Declare the gccloopsex1vec C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex1vec(MemRef<int, 1> *output,
                              MemRef<int, 1> *input1,
                              MemRef<int, 1> *input2);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRGccLoopsEx1Vec_1[1] = {10};
intptr_t sizesInputArrayMLIRGccLoopsEx1Vec_2[1] = {10};
intptr_t sizesOutputArrayMLIRGccLoopsEx1Vec[1] = {10};
// Define the MemRef container for inputs and output.
MemRef<int, 1> inputMLIRGccLoopsEx1Vec_1(sizesInputArrayMLIRGccLoopsEx1Vec_1, 2);
MemRef<int, 1> inputMLIRGccLoopsEx1Vec_2(sizesInputArrayMLIRGccLoopsEx1Vec_2, 3);
MemRef<int, 1> outputMLIRGccLoopsEx1Vec(sizesOutputArrayMLIRGccLoopsEx1Vec, 0);

static void MLIR_GccLoopsEx1Vec(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex1vec(&outputMLIRGccLoopsEx1Vec, &inputMLIRGccLoopsEx1Vec_1,
                               &inputMLIRGccLoopsEx1Vec_2);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx1Vec)->Arg(1);
// Generate result image.
void generateResultMLIRGccLoopsEx1Vec() {
  // Define the MemRef descriptor for inputs and output.
  MemRef<int, 1> input1(sizesInputArrayMLIRGccLoopsEx1Vec_1, 2);
  MemRef<int, 1> input2(sizesInputArrayMLIRGccLoopsEx1Vec_2, 3);
  MemRef<int, 1> output(sizesOutputArrayMLIRGccLoopsEx1Vec, 0);
  // Run the gccloopsex1vec.
  _mlir_ciface_mlir_gccloopsex1vec(&output, &input1, &input2);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx1Vec: MLIR GccLoopsEx1Vec Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
```

后面的步骤和问题1一样：

```
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DVECTORIZATION_BENCHMARKS=ON \
    -DBUDDY_MLIR_BUILD_DIR=/home/xxxxx/buddy-mlir/build   #这里要换成你自己编译好的buddy-mlir/build目录。

$ ninja vectorization-gccloops-benchmark

$ cd bin
$ ./vectorization-gccloops-benchmark
```

