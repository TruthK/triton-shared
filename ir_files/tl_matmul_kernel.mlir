#map = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
module {
  func.func @matmul_kernel(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {noinline = false} {
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %false = arith.constant false
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c63_i32 = arith.constant 63 : i32
    %0 = tensor.empty() : tensor<128x256xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %2 = tensor.empty() : tensor<64x256xf16>
    %3 = tensor.empty() : tensor<128x64xf16>
    %4 = tensor.empty() : tensor<128x64xi32>
    %5 = linalg.fill ins(%c64_i32 : i32) outs(%4 : tensor<128x64xi32>) -> tensor<128x64xi32>
    %6 = tt.get_program_id x : i32
    %7 = arith.addi %arg3, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.addi %arg4, %c255_i32 : i32
    %10 = arith.divsi %9, %c256_i32 : i32
    %11 = arith.muli %10, %c8_i32 : i32
    %12 = arith.divsi %6, %11 : i32
    %13 = arith.muli %12, %c8_i32 : i32
    %14 = arith.subi %8, %13 : i32
    %15 = arith.minsi %14, %c8_i32 : i32
    %16 = arith.remsi %6, %15 : i32
    %17 = arith.addi %13, %16 : i32
    %18 = arith.remsi %6, %11 : i32
    %19 = arith.divsi %18, %15 : i32
    %20 = arith.muli %17, %c128_i32 : i32
    %21 = tensor.empty() : tensor<128xi32>
    %22 = linalg_ext.make_range {operandSegmentSizes = array<i32: 2, 1>} ins(%c0_i32, %c128_i32 : i32, i32) outs(%21 : tensor<128xi32>) -> tensor<128xi32>
    %23 = linalg.fill ins(%20 : i32) outs(%21 : tensor<128xi32>) -> tensor<128xi32>
    %mapped = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%23, %22 : tensor<128xi32>, tensor<128xi32>) outs(%21 : tensor<128xi32>)
    %24 = linalg.fill ins(%arg3 : i32) outs(%21 : tensor<128xi32>) -> tensor<128xi32>
    %mapped_1 = linalg.map { arith.remsi } ins(%mapped, %24 : tensor<128xi32>, tensor<128xi32>) outs(%21 : tensor<128xi32>)
    %25 = arith.muli %19, %c256_i32 : i32
    %26 = tensor.empty() : tensor<256xi32>
    %27 = linalg_ext.make_range {operandSegmentSizes = array<i32: 2, 1>} ins(%c0_i32, %c256_i32 : i32, i32) outs(%26 : tensor<256xi32>) -> tensor<256xi32>
    %28 = linalg.fill ins(%25 : i32) outs(%26 : tensor<256xi32>) -> tensor<256xi32>
    %mapped_2 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%28, %27 : tensor<256xi32>, tensor<256xi32>) outs(%26 : tensor<256xi32>)
    %29 = linalg.fill ins(%arg4 : i32) outs(%26 : tensor<256xi32>) -> tensor<256xi32>
    %mapped_3 = linalg.map { arith.remsi } ins(%mapped_2, %29 : tensor<256xi32>, tensor<256xi32>) outs(%26 : tensor<256xi32>)
    %30 = tensor.empty() : tensor<64xi32>
    %31 = linalg_ext.make_range {operandSegmentSizes = array<i32: 2, 1>} ins(%c0_i32, %c64_i32 : i32, i32) outs(%30 : tensor<64xi32>) -> tensor<64xi32>
    %expanded = tensor.expand_shape %mapped_1 [[0, 1]] output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>
    %32 = tensor.empty() : tensor<128x1xi32>
    %33 = linalg.fill ins(%arg6 : i32) outs(%32 : tensor<128x1xi32>) -> tensor<128x1xi32>
    %mapped_4 = linalg.map { arith.muli {overflowFlags = #arith.overflow<none>} } ins(%expanded, %33 : tensor<128x1xi32>, tensor<128x1xi32>) outs(%32 : tensor<128x1xi32>)
    %collapsed = tensor.collapse_shape %mapped_4 [[0, 1]] : tensor<128x1xi32> into tensor<128xi32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<128xi32>) outs(%4 : tensor<128x64xi32>) dimensions = [1] 
    %broadcasted_5 = linalg.broadcast ins(%31 : tensor<64xi32>) outs(%4 : tensor<128x64xi32>) dimensions = [0] 
    %mapped_6 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%broadcasted, %broadcasted_5 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%4 : tensor<128x64xi32>)
    %expanded_7 = tensor.expand_shape %31 [[0, 1]] output_shape [64, 1] : tensor<64xi32> into tensor<64x1xi32>
    %34 = tensor.empty() : tensor<64x1xi32>
    %35 = linalg.fill ins(%arg7 : i32) outs(%34 : tensor<64x1xi32>) -> tensor<64x1xi32>
    %mapped_8 = linalg.map { arith.muli {overflowFlags = #arith.overflow<none>} } ins(%expanded_7, %35 : tensor<64x1xi32>, tensor<64x1xi32>) outs(%34 : tensor<64x1xi32>)
    %collapsed_9 = tensor.collapse_shape %mapped_8 [[0, 1]] : tensor<64x1xi32> into tensor<64xi32>
    %36 = tensor.empty() : tensor<64x256xi32>
    %broadcasted_10 = linalg.broadcast ins(%collapsed_9 : tensor<64xi32>) outs(%36 : tensor<64x256xi32>) dimensions = [1] 
    %broadcasted_11 = linalg.broadcast ins(%mapped_3 : tensor<256xi32>) outs(%36 : tensor<64x256xi32>) dimensions = [0] 
    %mapped_12 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%broadcasted_10, %broadcasted_11 : tensor<64x256xi32>, tensor<64x256xi32>) outs(%36 : tensor<64x256xi32>)
    %37 = arith.addi %arg5, %c63_i32 : i32
    %38 = arith.divsi %37, %c64_i32 : i32
    %39 = arith.muli %arg7, %c64_i32 : i32
    %40 = linalg.fill ins(%39 : i32) outs(%36 : tensor<64x256xi32>) -> tensor<64x256xi32>
    %41 = tensor.empty() : tensor<1x64xi1>
    %42 = tensor.empty() : tensor<128x64xi1>
    %43 = llvm.inttoptr %arg0 : i64 to !llvm.ptr
    %collapsed_13 = tensor.collapse_shape %3 [[0, 1]] : tensor<128x64xf16> into tensor<8192xf16>
    %expanded_14 = tensor.expand_shape %collapsed_13 [[0, 1]] output_shape [8192, 1] : tensor<8192xf16> into tensor<8192x1xf16>
    %44 = tensor.empty() : tensor<64x1xi1>
    %45 = tensor.empty() : tensor<64x256xi1>
    %46 = llvm.inttoptr %arg1 : i64 to !llvm.ptr
    %collapsed_15 = tensor.collapse_shape %2 [[0, 1]] : tensor<64x256xf16> into tensor<16384xf16>
    %expanded_16 = tensor.expand_shape %collapsed_15 [[0, 1]] output_shape [16384, 1] : tensor<16384xf16> into tensor<16384x1xf16>
    %47:3 = scf.for %arg9 = %c0_i32 to %38 step %c1_i32 iter_args(%arg10 = %1, %arg11 = %mapped_6, %arg12 = %mapped_12) -> (tensor<128x256xf32>, tensor<128x64xi32>, tensor<64x256xi32>)  : i32 {
      %74 = arith.muli %arg9, %c64_i32 : i32
      %75 = arith.subi %arg5, %74 : i32
      %76 = arith.index_cast %75 : i32 to index
      %77 = arith.maxsi %76, %c0 : index
      %78 = arith.minsi %77, %c64 : index
      %79 = tensor.empty(%78) : tensor<1x?xi1>
      %80 = linalg.fill ins(%true : i1) outs(%79 : tensor<1x?xi1>) -> tensor<1x?xi1>
      %81 = arith.subi %c64, %78 : index
      %82 = linalg_ext.pad ins(%80 : tensor<1x?xi1>) outs(%41 : tensor<1x64xi1>) pvalue(%false : i1) low = [0, 0] high = [0, %81] {
      ^bb0(%arg13: i1):
        linalg_ext.yield %arg13 : i1
      } -> tensor<1x64xi1>
      %collapsed_18 = tensor.collapse_shape %82 [[0, 1]] : tensor<1x64xi1> into tensor<64xi1>
      %broadcasted_19 = linalg.broadcast ins(%collapsed_18 : tensor<64xi1>) outs(%42 : tensor<128x64xi1>) dimensions = [0] 
      %83 = linalg.fill ins(%c0_i32 : i32) outs(%4 : tensor<128x64xi32>) -> tensor<128x64xi32>
      %mapped_20 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%arg11, %83 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%4 : tensor<128x64xi32>)
      %view_memref_21 = aux.view %43 to offset: [0], sizes: [9223372036854775807], strides: [1] : !llvm.ptr to memref<9223372036854775807xf16>
      %84 = bufferization.to_tensor %view_memref_21 restrict writable : memref<9223372036854775807xf16>
      %85 = linalg.fill ins(%cst : f16) outs(%expanded_14 : tensor<8192x1xf16>) -> tensor<8192x1xf16>
      %collapsed_22 = tensor.collapse_shape %mapped_20 [[0, 1]] : tensor<128x64xi32> into tensor<8192xi32>
      %expanded_23 = tensor.expand_shape %collapsed_22 [[0, 1]] output_shape [8192, 1] : tensor<8192xi32> into tensor<8192x1xi32>
      %collapsed_24 = tensor.collapse_shape %broadcasted_19 [[0, 1]] : tensor<128x64xi1> into tensor<8192xi1>
      %86 = linalg_ext.gather dimension_map = [0] ranged_data(false) signed_indice(true) ins(%84, %expanded_23, %collapsed_24 : tensor<9223372036854775807xf16>, tensor<8192x1xi32>, tensor<8192xi1>) outs(%85 : tensor<8192x1xf16>) {
      ^bb0(%arg13: f16, %arg14: f16):
        linalg_ext.yield %arg13 : f16
      } -> tensor<8192x1xf16>
      %collapsed_25 = tensor.collapse_shape %86 [[0, 1]] : tensor<8192x1xf16> into tensor<8192xf16>
      %expanded_26 = tensor.expand_shape %collapsed_25 [[0, 1]] output_shape [128, 64] : tensor<8192xf16> into tensor<128x64xf16>
      %87 = tensor.empty(%78) : tensor<?x1xi1>
      %88 = linalg.fill ins(%true : i1) outs(%87 : tensor<?x1xi1>) -> tensor<?x1xi1>
      %89 = linalg_ext.pad ins(%88 : tensor<?x1xi1>) outs(%44 : tensor<64x1xi1>) pvalue(%false : i1) low = [0, 0] high = [%81, 0] {
      ^bb0(%arg13: i1):
        linalg_ext.yield %arg13 : i1
      } -> tensor<64x1xi1>
      %collapsed_27 = tensor.collapse_shape %89 [[0, 1]] : tensor<64x1xi1> into tensor<64xi1>
      %broadcasted_28 = linalg.broadcast ins(%collapsed_27 : tensor<64xi1>) outs(%45 : tensor<64x256xi1>) dimensions = [1] 
      %90 = linalg.fill ins(%c0_i32 : i32) outs(%36 : tensor<64x256xi32>) -> tensor<64x256xi32>
      %mapped_29 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%arg12, %90 : tensor<64x256xi32>, tensor<64x256xi32>) outs(%36 : tensor<64x256xi32>)
      %view_memref_30 = aux.view %46 to offset: [0], sizes: [9223372036854775807], strides: [1] : !llvm.ptr to memref<9223372036854775807xf16>
      %91 = bufferization.to_tensor %view_memref_30 restrict writable : memref<9223372036854775807xf16>
      %92 = linalg.fill ins(%cst : f16) outs(%expanded_16 : tensor<16384x1xf16>) -> tensor<16384x1xf16>
      %collapsed_31 = tensor.collapse_shape %mapped_29 [[0, 1]] : tensor<64x256xi32> into tensor<16384xi32>
      %expanded_32 = tensor.expand_shape %collapsed_31 [[0, 1]] output_shape [16384, 1] : tensor<16384xi32> into tensor<16384x1xi32>
      %collapsed_33 = tensor.collapse_shape %broadcasted_28 [[0, 1]] : tensor<64x256xi1> into tensor<16384xi1>
      %93 = linalg_ext.gather dimension_map = [0] ranged_data(false) signed_indice(true) ins(%91, %expanded_32, %collapsed_33 : tensor<9223372036854775807xf16>, tensor<16384x1xi32>, tensor<16384xi1>) outs(%92 : tensor<16384x1xf16>) {
      ^bb0(%arg13: f16, %arg14: f16):
        linalg_ext.yield %arg13 : f16
      } -> tensor<16384x1xf16>
      %collapsed_34 = tensor.collapse_shape %93 [[0, 1]] : tensor<16384x1xf16> into tensor<16384xf16>
      %expanded_35 = tensor.expand_shape %collapsed_34 [[0, 1]] output_shape [64, 256] : tensor<16384xf16> into tensor<64x256xf16>
      %94 = linalg.matmul {__allow_tf32__} ins(%expanded_26, %expanded_35 : tensor<128x64xf16>, tensor<64x256xf16>) outs(%arg10 : tensor<128x256xf32>) -> tensor<128x256xf32>
      %mapped_36 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%arg11, %5 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%4 : tensor<128x64xi32>)
      %mapped_37 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%40, %arg12 : tensor<64x256xi32>, tensor<64x256xi32>) outs(%36 : tensor<64x256xi32>)
      scf.yield %94, %mapped_36, %mapped_37 : tensor<128x256xf32>, tensor<128x64xi32>, tensor<64x256xi32>
    }
    %48 = tensor.empty() : tensor<128x256xf16>
    %mapped_17 = linalg.map { arith.truncf } ins(%47#0 : tensor<128x256xf32>) outs(%48 : tensor<128x256xf16>)
    %49 = arith.addi %20, %c1_i32 : i32
    %50 = arith.muli %arg8, %49 : i32
    %51 = arith.muli %arg8, %20 : i32
    %52 = arith.addi %25, %50 : i32
    %53 = arith.addi %25, %51 : i32
    %54 = arith.index_cast %20 : i32 to index
    %55 = arith.addi %54, %c128 : index
    %56 = arith.index_cast %arg3 : i32 to index
    %57 = arith.maxsi %56, %54 : index
    %58 = arith.minsi %55, %57 : index
    %59 = arith.subi %58, %54 : index
    %60 = arith.index_cast %25 : i32 to index
    %61 = arith.addi %60, %c256 : index
    %62 = arith.index_cast %arg4 : i32 to index
    %63 = arith.maxsi %62, %60 : index
    %64 = arith.minsi %61, %63 : index
    %65 = arith.subi %64, %60 : index
    %66 = arith.minsi %59, %c128 : index
    %67 = arith.maxsi %66, %c0 : index
    %68 = arith.minsi %65, %c256 : index
    %69 = arith.maxsi %68, %c0 : index
    %70 = arith.subi %52, %53 : i32
    %71 = arith.index_cast %70 : i32 to index
    %72 = arith.index_cast %53 : i32 to index
    %73 = llvm.inttoptr %arg2 : i64 to !llvm.ptr
    %view_memref = aux.view %73 to offset: [%72], sizes: [%67, %69], strides: [%71, 1] : !llvm.ptr to memref<?x?xf16, #map>
    %extracted_slice = tensor.extract_slice %mapped_17[0, 0] [%67, %69] [1, 1] : tensor<128x256xf16> to tensor<?x?xf16>
    bufferization.materialize_in_destination %extracted_slice in writable %view_memref : (tensor<?x?xf16>, memref<?x?xf16, #map>) -> ()
    return
  }
}

