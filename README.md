# baseon-buddy-compiler
这个项目包含：

1.手把手教你构建buddy-compiler；

2.对buddy-benchmark作了一些改动，改动的动机来自于slack。

在看完装环境.md，并且按照步骤编译完所需的工具之后，可以看BaseQAndA中的三个问题，里面的三个子文件夹是对应的方案。你也可以下载本项目并运行查看结果。

下载本项目：

> **_ATTENTION:_**保证该项目和buddy-mlir在同一个目录下。

```
git clone git@github.com:YLChenZ/baseon-buddy-compiler.git
```
运行本项目，请参照下面的步骤：

Q1A：
```
$ cd baseon-buddy-compiler/BaseQAndA/Q1A
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_BUILD_DIR=/home/XXX/buddy-mlir/build  #这里要填你自己的buddy-mlir/build具体路径，最好是绝对路径
$ ninja vectorization-gccloops-benchmark
$ cd bin
$ ./vectorization-gccloops-benchmark
```

Q2A：
```
$ cd baseon-buddy-compiler/BaseQAndA/Q2A
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_BUILD_DIR=/home/XXX/buddy-mlir/build  #这里要填你自己的buddy-mlir/build具体路径，最好是绝对路径
$ ninja vectorization-gccloops-benchmark
$ cd bin
$ ./vectorization-gccloops-benchmark
```

在做Q3A之前，要保证[toolchain-riscv64.cmake的配置](https://github.com/YLChenZ/baseon-buddy-compiler/blob/main/BaseQAndA/Q3A/README.md#%E9%92%88%E5%AF%B9risc-v%E6%88%91%E5%8D%95%E7%8B%AC%E5%86%99%E4%BA%86%E4%B8%80%E4%B8%AA%E6%96%87%E4%BB%B6toolchain-riscv64cmake%E6%9D%A5%E6%8C%87%E5%AE%9A%E7%BC%96%E8%AF%91%E5%B7%A5%E5%85%B7%E9%93%BE)和[makefile的配置](https://github.com/YLChenZ/baseon-buddy-compiler/tree/main/BaseQAndA/Q3A#%E6%9C%80%E5%90%8E%E9%87%87%E7%94%A8qemu%E6%9D%A5%E6%A8%A1%E6%8B%9F%E6%89%A7%E8%A1%8C%E5%BE%97%E5%88%B0%E7%9A%84vectorbenchmark)是和你的机器工具环境适配。

具体参考：https://github.com/YLChenZ/baseon-buddy-compiler/tree/main/BaseQAndA/Q3A/README.md

Q3A：
```
$ cd baseon-buddy-compiler/BaseQAndA/Q3A
$ mkdir build && cd build
$ cmake -G Ninja ..  \
    -DCMAKE_BUILD_TYPE=RELEASE   \
    -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv64.cmake  \
    -DBUDDY_MLIR_BUILD_DIR=/home/XXX/buddy-mlir/build  #这里要填你自己的buddy-mlir/build具体路径，最好是绝对路径
$ ninja Vectorbenchmark
$ cd ..
$ make vector-benchmark-run
```


