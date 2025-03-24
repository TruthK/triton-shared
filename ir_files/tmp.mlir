func.func @matmul_kernel(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCoreMmaSync workgroup_size = [64, 2, 1] subgroup_size = 32, {pipeline_depth = 4 : i64, store_stage = 1 : i64}>} {
  %cst = arith.constant dense<0.000000e+00> : vector<2x2xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<16x8xf16>
  %cst_1 = arith.constant dense<0.000000e+00> : vector<16x8xf32>
  %c512 = arith.constant 512 : index
  %c2 = arith.constant 2 : index
  %c8_i32 = arith.constant 8 : i32
  %c128_i32 = arith.constant 128 : i32
  %c256_i32 = arith.constant 256 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i32 = arith.constant 1 : i32
  %c127_i32 = arith.constant 127 : i32
  %c255_i32 = arith.constant 255 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = arith.index_cast %c0_i32 : i32 to index
  %c256 = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c63_i32 = arith.constant 63 : i32
  %cst_2 = arith.constant 0.000000e+00 : f32
  %c32 = arith.constant 32 : index
  %alloc = memref.alloc() : memref<128x256xf16, #gpu.address_space<workgroup>>
  %alloc_3 = memref.alloc() : memref<128x256xf32, #gpu.address_space<workgroup>>
  %alloc_4 = memref.alloc() : memref<128x256xf32, #gpu.address_space<workgroup>>
  %alloc_5 = memref.alloc() : memref<4x128x32xf16, #gpu.address_space<workgroup>>
  %alloc_6 = memref.alloc() : memref<4x32x256xf16, #gpu.address_space<workgroup>>
  %alloc_7 = memref.alloc() : memref<128x64xf16, #gpu.address_space<workgroup>>
  %alloc_8 = memref.alloc() : memref<64x256xf16, #gpu.address_space<workgroup>>
  %thread_id_x = gpu.thread_id  x
  %thread_id_y = gpu.thread_id  y
  %1 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%thread_id_y]
  %2 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 16)>()[%thread_id_x]
  scf.for %arg15 = %1 to %c128 step %c32 {
    scf.for %arg16 = %2 to %c256 step %c32 {
      vector.transfer_write %cst_1, %alloc_4[%arg15, %arg16] {in_bounds = [true, true]} : vector<16x8xf32>, memref<128x256xf32, #gpu.address_space<workgroup>>
      %45 = affine.apply affine_map<()[s0] -> (s0 + 8)>()[%arg16]
      vector.transfer_write %cst_1, %alloc_4[%arg15, %45] {in_bounds = [true, true]} : vector<16x8xf32>, memref<128x256xf32, #gpu.address_space<workgroup>>
    }
  }
  %3 = arith.addi %arg3, %c127_i32 : i32
  %4 = arith.divsi %3, %c128_i32 : i32
  %5 = arith.addi %arg4, %c255_i32 : i32
  %6 = arith.divsi %5, %c256_i32 : i32
  %7 = arith.muli %6, %c8_i32 : i32
  %8 = arith.divsi %arg12, %7 : i32
  %9 = arith.muli %8, %c8_i32 : i32
  %10 = arith.subi %4, %9 : i32
  %11 = arith.minsi %10, %c8_i32 : i32
  %12 = arith.remsi %arg12, %11 : i32
  %13 = arith.addi %9, %12 : i32
  %14 = arith.remsi %arg12, %7 : i32
  %15 = arith.divsi %14, %11 : i32
  %16 = arith.muli %13, %c128_i32 : i32
  %17 = arith.index_cast %16 : i32 to index
  %18 = arith.muli %15, %c256_i32 : i32
  %19 = arith.index_cast %18 : i32 to index
  %20 = arith.index_cast %arg3 : i32 to index
  %21 = arith.index_cast %arg6 : i32 to index
  %22 = arith.muli %17, %21 : index
  %23 = arith.muli %20, %21 : index
  %24 = arith.index_cast %arg7 : i32 to index
  %25 = arith.index_cast %arg4 : i32 to index
  %26 = arith.addi %arg5, %c63_i32 : i32
  %27 = arith.divsi %26, %c64_i32 : i32
  %28 = arith.index_cast %27 : i32 to index
  %29 = arith.muli %arg7, %c64_i32 : i32
  %30 = arith.index_cast %29 : i32 to index
  %31 = arith.addi %19, %c256 : index
  %32 = arith.minsi %31, %25 : index
  %33:2 = scf.for %arg15 = %c0_i32 to %27 step %c1_i32 iter_args(%arg16 = %22, %arg17 = %c0) -> (index, index)  : i32 {
    %45 = arith.addi %arg17, %19 : index
    %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [64, 256], strides: [%24, %c1] : memref<*xf16> to memref<64x256xf16, strided<[?, ?], offset: ?>>
    %reinterpret_cast_10 = memref.reinterpret_cast %arg0 to offset: [%arg16], sizes: [128, 64], strides: [%21, %c1] : memref<*xf16> to memref<128x64xf16, strided<[?, ?], offset: ?>>
    %46 = arith.muli %arg15, %c64_i32 : i32
    %47 = arith.subi %arg5, %46 : i32
    %48 = arith.index_cast %47 : i32 to index
    %49 = arith.minsi %48, %c64 : index
    %50 = arith.maxsi %49, %c0 : index
    %51 = arith.addi %arg16, %c128 : index
    %52 = arith.minsi %51, %23 : index
    scf.for %arg18 = %1 to %c128 step %c32 {
      scf.for %arg19 = %2 to %c64 step %c32 {
        vector.transfer_write %cst_0, %alloc_7[%arg18, %arg19] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x64xf16, #gpu.address_space<workgroup>>
        %55 = affine.apply affine_map<()[s0] -> (s0 + 8)>()[%arg19]
        vector.transfer_write %cst_0, %alloc_7[%arg18, %55] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x64xf16, #gpu.address_space<workgroup>>
      }
    }
    gpu.barrier
    scf.for %arg18 = %thread_id_y to %52 step %c2 {
      %55 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%thread_id_x]
      scf.for %arg19 = %55 to %50 step %c512 {
        %56 = affine.min affine_map<(d0, d1) -> (8, d0 - d1)>(%50, %arg19)
        %subview = memref.subview %reinterpret_cast_10[%arg18, %arg19] [1, %56] [1, 1] : memref<128x64xf16, strided<[?, ?], offset: ?>> to memref<1x?xf16, strided<[?, ?], offset: ?>>
        %subview_11 = memref.subview %alloc_7[%arg18, %arg19] [1, %56] [1, 1] : memref<128x64xf16, #gpu.address_space<workgroup>> to memref<1x?xf16, strided<[64, 1], offset: ?>, #gpu.address_space<workgroup>>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<1x?xf16, strided<[?, ?], offset: ?>>) outs(%subview_11 : memref<1x?xf16, strided<[64, 1], offset: ?>, #gpu.address_space<workgroup>>) attrs =  {__internal_linalg_transform__ = "vectorize"} {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        }
      }
    }
    gpu.barrier
    scf.for %arg18 = %1 to %c64 step %c32 {
      scf.for %arg19 = %2 to %c256 step %c32 {
        vector.transfer_write %cst_0, %alloc_8[%arg18, %arg19] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x256xf16, #gpu.address_space<workgroup>>
        %55 = affine.apply affine_map<()[s0] -> (s0 + 8)>()[%arg19]
        vector.transfer_write %cst_0, %alloc_8[%arg18, %55] {in_bounds = [true, true]} : vector<16x8xf16>, memref<64x256xf16, #gpu.address_space<workgroup>>
      }
    }
    gpu.barrier
    scf.for %arg18 = %thread_id_y to %50 step %c2 {
      %55 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%thread_id_x]
      scf.for %arg19 = %55 to %32 step %c512 {
        %56 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 8)>(%arg19)[%32]
        %subview = memref.subview %reinterpret_cast_9[%arg18, %arg19] [1, %56] [1, 1] : memref<64x256xf16, strided<[?, ?], offset: ?>> to memref<1x?xf16, strided<[?, ?], offset: ?>>
        %subview_11 = memref.subview %alloc_8[%arg18, %arg19] [1, %56] [1, 1] : memref<64x256xf16, #gpu.address_space<workgroup>> to memref<1x?xf16, strided<[256, 1], offset: ?>, #gpu.address_space<workgroup>>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<1x?xf16, strided<[?, ?], offset: ?>>) outs(%subview_11 : memref<1x?xf16, strided<[256, 1], offset: ?>, #gpu.address_space<workgroup>>) attrs =  {__internal_linalg_transform__ = "vectorize"} {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        }
      }
    }
    gpu.barrier
    scf.for %arg18 = %1 to %c128 step %c32 {
      scf.for %arg19 = %2 to %c256 step %c32 {
        vector.transfer_write %cst_1, %alloc_3[%arg18, %arg19] {in_bounds = [true, true]} : vector<16x8xf32>, memref<128x256xf32, #gpu.address_space<workgroup>>
        %55 = affine.apply affine_map<()[s0] -> (s0 + 8)>()[%arg19]
        vector.transfer_write %cst_1, %alloc_3[%arg18, %55] {in_bounds = [true, true]} : vector<16x8xf32>, memref<128x256xf32, #gpu.address_space<workgroup>>
      }
    }
    scf.for %arg18 = %c0 to %c64 step %c32 {
      %55 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) mod 4)>(%arg18)
      gpu.barrier
      scf.for %arg19 = %thread_id_y to %c128 step %c2 {
        %56 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%thread_id_x]
        scf.for %arg20 = %56 to %c32 step %c512 {
          %57 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg18, %arg20]
          %subview = memref.subview %alloc_7[%arg19, %57] [1, 8] [1, 1] : memref<128x64xf16, #gpu.address_space<workgroup>> to memref<1x8xf16, strided<[64, 1], offset: ?>, #gpu.address_space<workgroup>>
          %subview_11 = memref.subview %alloc_5[%55, %arg19, %arg20] [1, 1, 8] [1, 1, 1] : memref<4x128x32xf16, #gpu.address_space<workgroup>> to memref<1x8xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<1x8xf16, strided<[64, 1], offset: ?>, #gpu.address_space<workgroup>>) outs(%subview_11 : memref<1x8xf16, strided<[32, 1], offset: ?>, #gpu.address_space<workgroup>>) attrs =  {__internal_linalg_transform__ = "vectorize"} {
          ^bb0(%in: f16, %out: f16):
            linalg.yield %in : f16
          }
        }
      }
      scf.for %arg19 = %thread_id_y to %c32 step %c2 {
        %56 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%thread_id_x]
        scf.for %arg20 = %56 to %c256 step %c512 {
          %57 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg18, %arg19]
          %subview = memref.subview %alloc_8[%57, %arg20] [1, 8] [1, 1] : memref<64x256xf16, #gpu.address_space<workgroup>> to memref<1x8xf16, strided<[256, 1], offset: ?>, #gpu.address_space<workgroup>>
          %subview_11 = memref.subview %alloc_6[%55, %arg19, %arg20] [1, 1, 8] [1, 1, 1] : memref<4x32x256xf16, #gpu.address_space<workgroup>> to memref<1x8xf16, strided<[256, 1], offset: ?>, #gpu.address_space<workgroup>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<1x8xf16, strided<[256, 1], offset: ?>, #gpu.address_space<workgroup>>) outs(%subview_11 : memref<1x8xf16, strided<[256, 1], offset: ?>, #gpu.address_space<workgroup>>) attrs =  {__internal_linalg_transform__ = "vectorize"} {
          ^bb0(%in: f16, %out: f16):
            linalg.yield %in : f16
          }
        }
      }
      gpu.barrier
      scf.for %arg19 = %1 to %c128 step %c32 {
        %56 = gpu.lane_id
        %57 = affine.apply affine_map<(d0)[s0] -> (d0 + s0 - (s0 floordiv 16) * 16)>(%arg19)[%56]
        %58 = affine.apply affine_map<()[s0] -> ((s0 floordiv 16) * 8)>()[%56]
        %59 = nvgpu.ldmatrix %alloc_5[%55, %57, %58] {numTiles = 4 : i32, transpose = false} : memref<4x128x32xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
        %60 = affine.apply affine_map<()[s0] -> ((s0 floordiv 16) * 8 + 16)>()[%56]
        %61 = nvgpu.ldmatrix %alloc_5[%55, %57, %60] {numTiles = 4 : i32, transpose = false} : memref<4x128x32xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
        scf.for %arg20 = %2 to %c256 step %c32 {
          %62 = affine.apply affine_map<(d0)[s0] -> (d0 + s0 floordiv 4)>(%arg19)[%56]
          %63 = affine.apply affine_map<(d0)[s0] -> (d0 + s0 * 2 - (s0 floordiv 4) * 8)>(%arg20)[%56]
          %64 = vector.load %alloc_3[%62, %63] : memref<128x256xf32, #gpu.address_space<workgroup>>, vector<2xf32>
          %65 = vector.insert %64, %cst [0] : vector<2xf32> into vector<2x2xf32>
          %66 = affine.apply affine_map<(d0)[s0] -> (d0 + s0 floordiv 4 + 8)>(%arg19)[%56]
          %67 = vector.load %alloc_3[%66, %63] : memref<128x256xf32, #gpu.address_space<workgroup>>, vector<2xf32>
          %68 = vector.insert %67, %65 [1] : vector<2xf32> into vector<2x2xf32>
          %69 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 2 - (s1 floordiv 4) * 8 + 8)>()[%arg20, %56]
          %70 = vector.load %alloc_3[%62, %69] : memref<128x256xf32, #gpu.address_space<workgroup>>, vector<2xf32>
          %71 = vector.insert %70, %cst [0] : vector<2xf32> into vector<2x2xf32>
          %72 = vector.load %alloc_3[%66, %69] : memref<128x256xf32, #gpu.address_space<workgroup>>, vector<2xf32>
          %73 = vector.insert %72, %71 [1] : vector<2xf32> into vector<2x2xf32>
          %74 = affine.apply affine_map<(d0)[s0] -> (d0 + (s0 floordiv 16) * 8)>(%arg20)[%56]
          %75 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%56]
          %76 = nvgpu.ldmatrix %alloc_6[%55, %75, %74] {numTiles = 4 : i32, transpose = true} : memref<4x32x256xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
          %77 = affine.apply affine_map<()[s0] -> (s0 mod 16 + 16)>()[%56]
          %78 = nvgpu.ldmatrix %alloc_6[%55, %77, %74] {numTiles = 4 : i32, transpose = true} : memref<4x32x256xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
          %79 = vector.extract_strided_slice %76 {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
          %80 = nvgpu.mma.sync(%59, %79, %68) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
          %81 = vector.extract_strided_slice %76 {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
          %82 = nvgpu.mma.sync(%59, %81, %73) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
          %83 = vector.extract_strided_slice %78 {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
          %84 = nvgpu.mma.sync(%61, %83, %80) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
          %85 = vector.extract_strided_slice %78 {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x2xf16> to vector<2x2xf16>
          %86 = nvgpu.mma.sync(%61, %85, %82) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
          %87 = vector.extract %84[0] : vector<2xf32> from vector<2x2xf32>
          vector.store %87, %alloc_3[%62, %63] : memref<128x256xf32, #gpu.address_space<workgroup>>, vector<2xf32>
          %88 = vector.extract %84[1] : vector<2xf32> from vector<2x2xf32>
          vector.store %88, %alloc_3[%66, %63] : memref<128x256xf32, #gpu.address_space<workgroup>>, vector<2xf32>
          %89 = vector.extract %86[0] : vector<2xf32> from vector<2x2xf32>
          vector.store %89, %alloc_3[%62, %69] : memref<128x256xf32, #gpu.address_space<workgroup>>, vector<2xf32>
          %90 = vector.extract %86[1] : vector<2xf32> from vector<2x2xf32>
          vector.store %90, %alloc_3[%66, %69] : memref<128x256xf32, #gpu.address_space<workgroup>>, vector<2xf32>
        }
      }
    }
    scf.for %arg18 = %1 to %c128 step %c32 {
      scf.for %arg19 = %2 to %c256 step %c32 {
        %55 = vector.transfer_read %alloc_4[%arg18, %arg19], %cst_2 {in_bounds = [true, true]} : memref<128x256xf32, #gpu.address_space<workgroup>>, vector<16x16xf32>
        %56 = vector.transfer_read %alloc_3[%arg18, %arg19], %cst_2 {in_bounds = [true, true]} : memref<128x256xf32, #gpu.address_space<workgroup>>, vector<16x16xf32>
        %57 = arith.addf %55, %56 {dot_c} : vector<16x16xf32>
        %58 = vector.extract_strided_slice %57 {offsets = [0, 0], sizes = [16, 8], strides = [1, 1]} : vector<16x16xf32> to vector<16x8xf32>
        vector.transfer_write %58, %alloc_4[%arg18, %arg19] {in_bounds = [true, true]} : vector<16x8xf32>, memref<128x256xf32, #gpu.address_space<workgroup>>
        %59 = vector.extract_strided_slice %57 {offsets = [0, 8], sizes = [16, 8], strides = [1, 1]} : vector<16x16xf32> to vector<16x8xf32>
        %60 = affine.apply affine_map<()[s0] -> (s0 + 8)>()[%arg19]
        vector.transfer_write %59, %alloc_4[%arg18, %60] {in_bounds = [true, true]} : vector<16x8xf32>, memref<128x256xf32, #gpu.address_space<workgroup>>
      }
    }
    %53 = arith.addi %arg16, %c64 : index
    %54 = arith.addi %arg17, %30 : index
    scf.yield %53, %54 : index, index
  }
  scf.forall (%arg15, %arg16) = (0, 0) to (128, 256) step (32, 32) {
    scf.for %arg17 = %1 to %c32 step %c32 {
      %45 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg15, %arg17]
      scf.for %arg18 = %2 to %c32 step %c32 {
        %46 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg16, %arg18]
        %47 = vector.transfer_read %alloc_4[%45, %46], %cst_2 {in_bounds = [true, true]} : memref<128x256xf32, #gpu.address_space<workgroup>>, vector<16x16xf32>
        %48 = arith.truncf %47 : vector<16x16xf32> to vector<16x16xf16>
        %49 = vector.extract_strided_slice %48 {offsets = [0, 0], sizes = [16, 8], strides = [1, 1]} : vector<16x16xf16> to vector<16x8xf16>
        vector.transfer_write %49, %alloc[%45, %46] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x256xf16, #gpu.address_space<workgroup>>
        %50 = vector.extract_strided_slice %48 {offsets = [0, 8], sizes = [16, 8], strides = [1, 1]} : vector<16x16xf16> to vector<16x8xf16>
        %51 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 8)>()[%arg16, %arg18]
        vector.transfer_write %50, %alloc[%45, %51] {in_bounds = [true, true]} : vector<16x8xf16>, memref<128x256xf16, #gpu.address_space<workgroup>>
      }
    }
    gpu.barrier
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  %34 = arith.index_cast %arg8 : i32 to index
  %35 = arith.muli %17, %34 : index
  %36 = arith.addi %35, %19 : index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [128, 256], strides: [%34, 1] : memref<*xf16> to memref<128x256xf16, strided<[?, 1], offset: ?>>
  %37 = arith.addi %17, %c128 : index
  %38 = arith.minsi %37, %20 : index
  %39 = arith.maxsi %38, %17 : index
  %40 = arith.subi %39, %17 : index
  %41 = arith.maxsi %32, %19 : index
  %42 = arith.subi %41, %19 : index
  %43 = arith.minsi %40, %c128 : index
  %44 = arith.minsi %42, %c256 : index
  gpu.barrier
  scf.for %arg15 = %thread_id_y to %43 step %c2 {
    %45 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%thread_id_x]
    scf.for %arg16 = %45 to %44 step %c512 {
      %46 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 8)>(%arg16)[%44]
      %subview = memref.subview %alloc[%arg15, %arg16] [1, %46] [1, 1] : memref<128x256xf16, #gpu.address_space<workgroup>> to memref<1x?xf16, strided<[256, 1], offset: ?>, #gpu.address_space<workgroup>>
      %subview_9 = memref.subview %reinterpret_cast[%arg15, %arg16] [1, %46] [1, 1] : memref<128x256xf16, strided<[?, 1], offset: ?>> to memref<1x?xf16, strided<[?, 1], offset: ?>>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<1x?xf16, strided<[256, 1], offset: ?>, #gpu.address_space<workgroup>>) outs(%subview_9 : memref<1x?xf16, strided<[?, 1], offset: ?>>) attrs =  {__internal_linalg_transform__ = "vectorize"} {
      ^bb0(%in: f16, %out: f16):
        linalg.yield %in : f16
      }
    }
  }
  gpu.barrier
  return
}