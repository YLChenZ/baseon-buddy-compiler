#!/bin/bash

#configure
CROSS_MLIR_LIB := ../../../buddy-mlir/llvm/build-cross-mlir-rv/lib
QEMU := ../../../buddy-mlir/build/thirdparty/riscv-gnu-toolchain/bin/qemu-riscv64
RISCV_GNU_TOOLCHAIN_SYSROOT := ../../../buddy-mlir/build/thirdparty/riscv-gnu-toolchain/sysroot/


vector-benchmark-run:
	@LD_LIBRARY_PATH=${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu max ./build/bin/Vectorbenchmark


