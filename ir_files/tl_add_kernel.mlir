#map = affine_map<(d0)[s0] -> (d0 + s0)>
module {
  func.func @add_kernel(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i32) attributes {noinline = false} {
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.addi %2, %c1024 : index
    %4 = arith.index_cast %arg3 : i32 to index
    %5 = arith.maxsi %4, %2 : index
    %6 = arith.minsi %3, %5 : index
    %7 = arith.subi %6, %2 : index
    %8 = llvm.inttoptr %arg0 : i64 to !llvm.ptr
    %view_memref = aux.view %8 to offset: [%2], sizes: [%7], strides: [1] : !llvm.ptr to memref<?xf32, #map>
    %9 = bufferization.to_tensor %view_memref restrict writable : memref<?xf32, #map>
    %10 = tensor.empty(%7) : tensor<?xf32>
    %11 = linalg.copy ins(%9 : tensor<?xf32>) outs(%10 : tensor<?xf32>) -> tensor<?xf32>
    %12 = tensor.empty() : tensor<1024xf32>
    %13 = arith.subi %c1024, %7 : index
    %14 = linalg_ext.pad ins(%11 : tensor<?xf32>) outs(%12 : tensor<1024xf32>) pvalue(%cst : f32) low = [0] high = [%13] {
    ^bb0(%arg4: f32):
      linalg_ext.yield %arg4 : f32
    } -> tensor<1024xf32>
    %15 = llvm.inttoptr %arg1 : i64 to !llvm.ptr
    %view_memref_0 = aux.view %15 to offset: [%2], sizes: [%7], strides: [1] : !llvm.ptr to memref<?xf32, #map>
    %16 = bufferization.to_tensor %view_memref_0 restrict writable : memref<?xf32, #map>
    %17 = linalg.copy ins(%16 : tensor<?xf32>) outs(%10 : tensor<?xf32>) -> tensor<?xf32>
    %18 = linalg_ext.pad ins(%17 : tensor<?xf32>) outs(%12 : tensor<1024xf32>) pvalue(%cst : f32) low = [0] high = [%13] {
    ^bb0(%arg4: f32):
      linalg_ext.yield %arg4 : f32
    } -> tensor<1024xf32>
    %mapped = linalg.map { arith.addf } ins(%14, %18 : tensor<1024xf32>, tensor<1024xf32>) outs(%12 : tensor<1024xf32>)
    %19 = llvm.inttoptr %arg2 : i64 to !llvm.ptr
    %view_memref_1 = aux.view %19 to offset: [%2], sizes: [%7], strides: [1] : !llvm.ptr to memref<?xf32, #map>
    %extracted_slice = tensor.extract_slice %mapped[0] [%7] [1] : tensor<1024xf32> to tensor<?xf32>
    bufferization.materialize_in_destination %extracted_slice in writable %view_memref_1 : (tensor<?xf32>, memref<?xf32, #map>) -> ()
    return
  }
}

