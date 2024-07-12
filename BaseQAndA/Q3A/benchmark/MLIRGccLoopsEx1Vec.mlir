//===----------------------------------------------------------------------===//
//
// This file provides the MLIR GccLoopsEx1Vec function.
//
//===----------------------------------------------------------------------===//


func.func @mlir_gccloopsex1vec(%A: memref<?xi32>, %B: memref<?xi32>,
                      %C: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %vector_size = arith.constant 4 : index  // 假设向量大小为4
  %c1 = arith.constant 1 : index
  %n = memref.dim %B, %c0 : memref<?xi32>
  
  //get upper bound
  %rem = arith.remsi %n, %vector_size : index
  %upper_bound = arith.subi %n, %rem : index
    // 向量化循环
  scf.for %i = %c0 to %upper_bound step %vector_size {
   // 加载向量
   %b_vec = vector.load %B[%i] : memref<?xi32>, vector<4xi32>
   %c_vec = vector.load %C[%i] : memref<?xi32>, vector<4xi32>
    // 元素相加
   %result_vec = arith.addi %b_vec, %c_vec : vector<4xi32>
    // 存储向量
   vector.store %result_vec, %A[%i] : memref<?xi32>, vector<4xi32>  
}

  // 处理剩余的元素
  
  scf.for %i = %upper_bound to %n step %c1 {
    %b_elem = memref.load %B[%i] : memref<?xi32>
    %c_elem = memref.load %C[%i] : memref<?xi32>
    %result_elem = arith.addi %b_elem, %c_elem : i32
    memref.store %result_elem, %A[%i] : memref<?xi32>
  }
  return
}
