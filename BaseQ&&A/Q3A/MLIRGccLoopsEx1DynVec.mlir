//===----------------------------------------------------------------------===//
//
// This file provides the MLIR GccLoopsEx1DynVec function.
//
//===----------------------------------------------------------------------===//

#map = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>

func.func @mlir_gccloopsex1dynvec(%A: memref<?xi32>, %B: memref<?xi32>,
                      %C: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %n = memref.dim %A, %c0 : memref<?xi32>
  // Perform dynamic vector addition.
  // Returns four times the physical vl for element type i32.
  %vl = vector_exp.get_vl i32, 4 : index

  scf.for %i = %c0 to %n step %vl { // Tiling
    %it_vl = affine.min #map(%i)[%vl, %n]
    vector_exp.set_vl %it_vl : index {
      %vec_b = vector.load %B[%i] : memref<?xi32>, vector<[1]xi32> // vector<?xi32>
      %vec_c = vector.load %C[%i] : memref<?xi32>, vector<[1]xi32> // vector<?xi32>
      %vec_res = arith.addi %vec_b, %vec_c : vector<[1]xi32> // vector<?xi32>
      vector.store %vec_res, %A[%i] : memref<?xi32>, vector<[1]xi32> // vector<?xi32>
      vector.yield
    }
  }
  return
}
