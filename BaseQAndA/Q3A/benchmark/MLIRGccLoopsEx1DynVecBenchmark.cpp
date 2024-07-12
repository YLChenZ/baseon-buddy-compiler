//===- MLIRGccLoopsEx1DynVecBenchmark.cpp --------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the benchmark for buddy-opt tool in buddy-mlir project.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
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
  std::cout << "MLIR_GccLoopsEx1DynVec: MLIR GccLoopsEx1DynVec Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
