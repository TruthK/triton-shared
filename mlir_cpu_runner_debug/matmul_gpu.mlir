module {
  func.func private @print_memref_f32(memref<32x32xf32>)
  func.func private @print_i32(index)
  func.func @main() {
    %A = memref.alloc() : memref<128x32xf32>
    %B = memref.alloc() : memref<32x128xf32>
    %C = memref.alloc() : memref<32x32xf32>

    %cst = arith.constant 2.1 : f32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index

    // Initialize A (128x32)
    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        memref.store %cst, %A[%i, %j] : memref<128x32xf32>
      }
    }

    // Initialize B (32x128)
    scf.for %i = %c0 to %c32 step %c1 {
      scf.for %j = %c0 to %c128 step %c1 {
        memref.store %cst, %B[%i, %j] : memref<32x128xf32>
      }
    }

    // Initialize C (32x32)
    scf.for %i = %c0 to %c32 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        memref.store %cst, %C[%i, %j] : memref<32x32xf32>
      }
    }

    %A_cast = memref.cast %A : memref<128x32xf32> to memref<*xf32>
    %B_cast = memref.cast %B : memref<32x128xf32> to memref<*xf32>
    %C_cast = memref.cast %C : memref<32x32xf32> to memref<*xf32>

    call @mma(%A_cast, %B_cast, %C_cast) : (memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()

    return
  }


func.func @mma(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}) {
    %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32_i32 = arith.constant 32 : i32
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c8 = arith.constant 8 : index
  %thread_id_x = arith.constant 0 : index
  %thread_id_y = arith.constant 0 : index
  %0 = affine.linearize_index disjoint [%thread_id_y, %thread_id_x] by (2, 64) : index
  %alloc = memref.alloc() : memref<32x32xf32>
  %alloc_0 = memref.alloc() : memref<4x32x16xf32>
  %alloc_1 = memref.alloc() : memref<4x16x32xf32>
  %block_id_x = arith.constant 0 : index
  %1 = arith.index_cast %block_id_x : index to i32
  %block_id_y = arith.constant 0 : index
  %2 = arith.index_cast %block_id_y : index to i32
  %3 = arith.muli %1, %c32_i32 : i32
  %4 = arith.index_cast %3 : i32 to index
  %5 = arith.muli %2, %c32_i32 : i32
  %6 = arith.index_cast %5 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%6], sizes: [128, 32], strides: [%c128, 1] : memref<*xf32> to memref<128x32xf32, strided<[?, 1], offset: ?>>
  %7 = arith.muli %4, %c128 : index
  %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 128], strides: [%c128, 1] : memref<*xf32> to memref<32x128xf32, strided<[?, 1], offset: ?>>
  %8 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%thread_id_y]
  %9 = affine.apply affine_map<()[s0] -> ((s0 floordiv 32) * 16)>()[%thread_id_x]
  scf.for %arg3 = %c0 to %c16 step %c16 {
    %17 = affine.apply affine_map<(d0) -> ((d0 floordiv 16) mod 4)>(%arg3)
    
    %18:2 = affine.delinearize_index %0 into (32, 4) : index, index
    %19 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 4)>()[%arg3, %18#1]
    %20 = vector.transfer_read %reinterpret_cast_2[%18#0, %19], %cst {in_bounds = [true, true]} : memref<32x128xf32, strided<[?, 1], offset: ?>>, vector<1x4xf32>
    %21 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%18#1]
    vector.transfer_write %20, %alloc_0[%17, %18#0, %21] {in_bounds = [true, true]} : vector<1x4xf32>, memref<4x32x16xf32>
    %22:2 = affine.delinearize_index %0 into (16, 8) : index, index
    %23 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg3, %22#0]
    %24 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%22#1]
    %25 = vector.transfer_read %reinterpret_cast[%23, %24], %cst {in_bounds = [true, true]} : memref<128x32xf32, strided<[?, 1], offset: ?>>, vector<1x4xf32>
    vector.transfer_write %25, %alloc_1[%17, %22#0, %24] {in_bounds = [true, true]} : vector<1x4xf32>, memref<4x16x32xf32>
    
    scf.for %arg4 = %8 to %c32 step %c32 {
      %26 = vector.transfer_read %alloc_0[%17, %arg4, %c0], %cst {in_bounds = [true, true]} : memref<4x32x16xf32>, vector<16x8xf32>
      %27 = vector.transfer_read %alloc_0[%17, %arg4, %c8], %cst {in_bounds = [true, true]} : memref<4x32x16xf32>, vector<16x8xf32>
      scf.for %arg5 = %9 to %c32 step %c32 {
        %28 = vector.transfer_read %alloc[%arg4, %arg5], %cst {in_bounds = [true, true]} : memref<32x32xf32>, vector<16x8xf32>
        %29 = affine.apply affine_map<()[s0] -> (s0 + 8)>()[%arg5]
        %30 = vector.transfer_read %alloc[%arg4, %29], %cst {in_bounds = [true, true]} : memref<32x32xf32>, vector<16x8xf32>
        %31 = vector.transfer_read %alloc_1[%17, %c0, %arg5], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2) -> (d2, d1)>} : memref<4x16x32xf32>, vector<8x8xf32>
        %32 = vector.transfer_read %alloc_1[%17, %c8, %arg5], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2) -> (d2, d1)>} : memref<4x16x32xf32>, vector<8x8xf32>
        %33 = vector.transfer_read %alloc_1[%17, %c0, %29], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2) -> (d2, d1)>} : memref<4x16x32xf32>, vector<8x8xf32>
        %34 = vector.transfer_read %alloc_1[%17, %c8, %29], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2) -> (d2, d1)>} : memref<4x16x32xf32>, vector<8x8xf32>
        %35 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %26, %31, %28 : vector<16x8xf32>, vector<8x8xf32> into vector<16x8xf32>
        %36 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %26, %33, %30 : vector<16x8xf32>, vector<8x8xf32> into vector<16x8xf32>
        %37 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %27, %32, %35 : vector<16x8xf32>, vector<8x8xf32> into vector<16x8xf32>
        %38 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %27, %34, %36 : vector<16x8xf32>, vector<8x8xf32> into vector<16x8xf32>
        vector.transfer_write %37, %alloc[%arg4, %arg5] {in_bounds = [true, true]} : vector<16x8xf32>, memref<32x32xf32>
        vector.transfer_write %38, %alloc[%arg4, %29] {in_bounds = [true, true]} : vector<16x8xf32>, memref<32x32xf32>
      }
    }
  }
  %10 = arith.muli %4, %c128 : index
  %11 = arith.addi %10, %6 : index
  %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [%c128, 1] : memref<*xf32> to memref<32x32xf32, strided<[?, 1], offset: ?>>
  
  %12:2 = affine.delinearize_index %0 into (16, 8) : index, index
  %13 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%12#1]
  %14 = vector.transfer_read %alloc[%12#0, %13], %cst {in_bounds = [true, true]} : memref<32x32xf32>, vector<1x4xf32>
  vector.transfer_write %14, %reinterpret_cast_3[%12#0, %13] {in_bounds = [true, true]} : vector<1x4xf32>, memref<32x32xf32, strided<[?, 1], offset: ?>>
  %15 = affine.apply affine_map<()[s0] -> (s0 + 16)>()[%12#0]
  %16 = vector.transfer_read %alloc[%15, %13], %cst {in_bounds = [true, true]} : memref<32x32xf32>, vector<1x4xf32>
  vector.transfer_write %16, %reinterpret_cast_3[%15, %13] {in_bounds = [true, true]} : vector<1x4xf32>, memref<32x32xf32, strided<[?, 1], offset: ?>>
  return
}
}