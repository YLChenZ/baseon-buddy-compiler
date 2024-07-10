编译成功`buddy-mlir`，然后参考`buddy-benchmark`的`README.md`，跑通`benchmarks/Vectorization`里面的`gccloops`（即，能成功运行`buddy-benchmark/build/bin`里的`vectorization-gccloops-benchmark`可执行文件）。该`benchmark`来自`LLVM`用于探究自动向量化性能的`benchmark`，具体可以参见

   [`LLVM`的自动向量化文档]: https://llvm.org/docs/Vectorizers.html#the-loop-vectorizer

   ，`gcc-loops`源代码在

   [此处]: https://github.com/llvm/llvm-test-suite/tree/main/SingleSource/UnitTests/Vectorize

   。

问题1还是比较基础的，按照官网步骤来：

```
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DVECTORIZATION_BENCHMARKS=ON \
    -DBUDDY_MLIR_BUILD_DIR=/home/xxx/buddy-mlir/build
    //这里要换成你自己编译好的buddy-mlir/build目录
$ ninja vectorization-gccloops-benchmark
```

此时在/bin目录下会有`vectorization-gccloops-benchmark`，执行它就行了。

```
$ cd bin
$ ./vectorization-gccloops-benchmark
```