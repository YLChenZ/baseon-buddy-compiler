
//===----------------------------------------------------------------------===//
//
// This is the main file of the gccloops vectorization benchmark.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>

void generateResultMLIRGccLoopsEx1();
void generateResultMLIRGccLoopsEx1Vec();
void generateResultMLIRGccLoopsEx1DynVec();
// Run benchmarks.
int main(int argc, char **argv) {

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  // Generate result.
  generateResultMLIRGccLoopsEx1();
  generateResultMLIRGccLoopsEx1Vec();
  generateResultMLIRGccLoopsEx1DynVec();
  return 0;
}
