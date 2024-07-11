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

Q3A：
```
$ cd baseon-buddy-compiler/BaseQAndA/Q3A
$ mkdir build && cd build
$ cmake -G Ninja ..
    -DCMAKE_BUILD_TYPE=RELEASE   \
    -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv64.cmake  \
    -DBUDDY_MLIR_BUILD_DIR=/home/XXX/buddy-mlir/build  #这里要填你自己的buddy-mlir/build具体路径，最好是绝对路径
$ ninja Vectorbenchmark
$ cd ..
$ make vector-benchmark-run
```


