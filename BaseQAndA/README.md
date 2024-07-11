1. 编译成功`buddy-mlir`，然后参考`buddy-benchmark`的`README.md`，跑通`benchmarks/Vectorization`里面的`gccloops`（即，能成功运行`buddy-benchmark/build/bin`里的`vectorization-gccloops-benchmark`可执行文件）。该`benchmark`来自`LLVM`用于探究自动向量化性能的`benchmark`，具体可以参见[LLVM的自动向量化文档](https://llvm.org/docs/Vectorizers.html#the-loop-vectorizer)，`gcc-loops`源代码在[此处](https://github.com/llvm/llvm-test-suite/tree/main/SingleSource/UnitTests/Vectorize)。

2. 查看上一个任务里`gcc-loops`的`MLIR`实现代码，以第一个程序`MLIRGccLoopsEx1.mlir`为例，可以看到它只是朴素的标量版本（用了`scf.for`，`arith.addi`每次只对一位进行操作）。请用`MLIR`的`Vector Dialect`实现一个同样功能的向量化版本，文件命名为`MLIRGccLoopsEx1Vec.mlir`，并加入到`buddy-benchmark`里，与`MLIRGccLoopsEx1.mlir`的性能进行比较。`Vector Dialect`的用法可参考`MLIR`官方文档或`buddy-compiler`中的已有实现，如`buddy-mlir`里的`MLIRVector`样例和`buddy-benchmark`里的矩阵乘法向量化实现。

3. 在实现上一个任务的向量化版本时，你是如何确定向量化长度的（即，一次取多少个元素进行存取和计算操作）？请带着这个问题，参考`buddy-compiler`的动态向量支持文档和向量加法实现，实现一个与`gcc-loops`程序具有同样功能的动态类型向量化版本，文件命名为`MLIRGccLoopsEx1DynVec.mlir`，然后面向`RISC-V`平台下降得到汇编指令，并与上一个任务的向量化版本的相比较。
