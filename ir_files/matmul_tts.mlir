#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @matmul_kernel(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c63_i32 = arith.constant 63 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c0_i32 = arith.constant 0 : i32
    %c255_i32 = arith.constant 255 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tensor.empty() : tensor<128x256xf32>
    %1 = linalg.fill {triton_ptr = #tts.triton_ptr} ins(%cst : f32) outs(%0 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %2 = arith.addi %arg3, %c127_i32 : i32
    %3 = arith.divsi %2, %c128_i32 : i32
    %4 = arith.addi %arg4, %c255_i32 : i32
    %5 = arith.divsi %4, %c256_i32 : i32
    %6 = arith.muli %5, %c8_i32 : i32
    %7 = arith.divsi %arg12, %6 : i32
    %8 = arith.muli %7, %c8_i32 : i32
    %9 = arith.subi %3, %8 : i32
    %10 = arith.minsi %9, %c8_i32 : i32
    %11 = arith.remsi %arg12, %10 : i32
    %12 = arith.addi %8, %11 : i32
    %13 = arith.remsi %arg12, %6 : i32
    %14 = arith.divsi %13, %10 : i32
    %15 = arith.muli %12, %c128_i32 : i32
    %16 = arith.index_cast %15 : i32 to index
    %17 = arith.muli %14, %c256_i32 : i32
    %18 = arith.index_cast %17 : i32 to index
    %19 = arith.index_cast %arg3 : i32 to index
    %20 = arith.index_cast %arg6 : i32 to index
    %21 = arith.muli %16, %20 : index
    %22 = arith.muli %19, %20 : index
    %23 = arith.index_cast %arg7 : i32 to index
    %24 = arith.index_cast %arg4 : i32 to index
    %25 = arith.addi %arg5, %c63_i32 : i32
    %26 = arith.divsi %25, %c64_i32 : i32
    %27 = arith.muli %arg7, %c64_i32 : i32
    %28 = arith.index_cast %27 : i32 to index
    %29:3 = scf.for %arg15 = %c0_i32 to %26 step %c1_i32 iter_args(%arg16 = %1, %arg17 = %21, %arg18 = %c0) -> (tensor<128x256xf32>, index, index)  : i32 {
      %45 = arith.addi %arg18, %18 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [64, 256], strides: [%23, %c1] : memref<*xf16> to memref<64x256xf16, strided<[?, ?], offset: ?>>
      %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%arg17], sizes: [128, 64], strides: [%20, %c1] : memref<*xf16> to memref<128x64xf16, strided<[?, ?], offset: ?>>
      %46 = arith.muli %arg15, %c64_i32 : i32
      %47 = arith.subi %arg5, %46 : i32
      %48 = arith.index_cast %47 : i32 to index
      %49 = arith.minsi %48, %c64 : index
      %50 = arith.maxsi %49, %c0 : index
      %51 = arith.addi %arg17, %c128 : index
      %52 = arith.minsi %51, %22 : index
      %53 = tensor.empty() : tensor<128x64xf16>
      %54 = linalg.fill {triton_ptr = #tts.triton_ptr} ins(%cst_0 : f16) outs(%53 : tensor<128x64xf16>) -> tensor<128x64xf16>
      %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%52, %50] [1, 1] : memref<128x64xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
      %55 = bufferization.to_tensor %subview_3 restrict : memref<?x?xf16, strided<[?, ?], offset: ?>> to tensor<?x?xf16>
      %inserted_slice = tensor.insert_slice %55 into %54[0, 0] [%52, %50] [1, 1] : tensor<?x?xf16> into tensor<128x64xf16>
      %56 = arith.addi %18, %c256 : index
      %57 = arith.minsi %56, %24 : index
      %58 = tensor.empty() : tensor<64x256xf16>
      %59 = linalg.fill {triton_ptr = #tts.triton_ptr} ins(%cst_0 : f16) outs(%58 : tensor<64x256xf16>) -> tensor<64x256xf16>
      %subview_4 = memref.subview %reinterpret_cast_1[0, 0] [%50, %57] [1, 1] : memref<64x256xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
      %60 = bufferization.to_tensor %subview_4 restrict : memref<?x?xf16, strided<[?, ?], offset: ?>> to tensor<?x?xf16>
      %inserted_slice_5 = tensor.insert_slice %60 into %59[0, 0] [%50, %57] [1, 1] : tensor<?x?xf16> into tensor<64x256xf16>
      %61 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x256xf32>) -> tensor<128x256xf32>
      %62 = linalg.matmul ins(%inserted_slice, %inserted_slice_5 : tensor<128x64xf16>, tensor<64x256xf16>) outs(%61 : tensor<128x256xf32>) -> tensor<128x256xf32>
      %63 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg16, %62 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%0 : tensor<128x256xf32>) {
      ^bb0(%in: f32, %in_6: f32, %out: f32):
        %66 = arith.addf %in, %in_6 : f32
        linalg.yield %66 : f32
      } -> tensor<128x256xf32>
      %64 = arith.addi %arg17, %c64 : index
      %65 = arith.addi %arg18, %28 : index
      scf.yield %63, %64, %65 : tensor<128x256xf32>, index, index
    }
    %30 = tensor.empty() : tensor<128x256xf16>
    %31 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%29#0 : tensor<128x256xf32>) outs(%30 : tensor<128x256xf16>) {
    ^bb0(%in: f32, %out: f16):
      %45 = arith.truncf %in : f32 to f16
      linalg.yield %45 : f16
    } -> tensor<128x256xf16>
    %32 = arith.index_cast %arg8 : i32 to index
    %33 = arith.muli %16, %32 : index
    %34 = arith.addi %33, %18 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [128, 256], strides: [%32, 1] : memref<*xf16> to memref<128x256xf16, strided<[?, 1], offset: ?>>
    %35 = arith.addi %16, %c128 : index
    %36 = arith.minsi %35, %19 : index
    %37 = arith.maxsi %36, %16 : index
    %38 = arith.subi %37, %16 : index
    %39 = arith.addi %18, %c256 : index
    %40 = arith.minsi %39, %24 : index
    %41 = arith.maxsi %40, %18 : index
    %42 = arith.subi %41, %18 : index
    %43 = arith.minsi %38, %c128 : index
    %44 = arith.minsi %42, %c256 : index
    %extracted_slice = tensor.extract_slice %31[0, 0] [%43, %44] [1, 1] : tensor<128x256xf16> to tensor<?x?xf16>
    %subview = memref.subview %reinterpret_cast[0, 0] [%43, %44] [1, 1] : memref<128x256xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>) -> ()
    return
  }
}

