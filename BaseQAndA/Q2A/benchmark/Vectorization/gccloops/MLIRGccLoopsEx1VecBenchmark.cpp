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

