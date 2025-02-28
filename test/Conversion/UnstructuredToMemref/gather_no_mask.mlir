// RUN: triton-shared-opt --triton-to-unstructured --canonicalize --unstructured-to-memref --canonicalize %s | FileCheck %s

module {
  tt.func public @gather_simple_no_mask(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<64> : tensor<64xi32>
    %c64_i32 = arith.constant 64 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_0 = arith.constant dense<10> : tensor<64xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %3:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %0, %arg4 = %0) -> (tensor<64xi32>, tensor<64xi32>)  : i32 {
      %4 = arith.divsi %arg3, %cst_0 : tensor<64xi32>
      %5 = arith.addi %arg2, %c5_i32 : i32
      %6 = arith.remsi %5, %c64_i32 : i32
      %7 = tt.splat %6 : i32 -> tensor<64xi32>
      %8 = arith.addi %4, %7 : tensor<64xi32>
      %9 = tt.addptr %1, %8 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %10 = tt.load %9 : tensor<64x!tt.ptr<f32>>
      %11 = tt.addptr %2, %arg4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      tt.store %11, %10 : tensor<64x!tt.ptr<f32>>
      %12 = arith.addi %8, %cst : tensor<64xi32>
      %13 = arith.addi %arg4, %cst : tensor<64xi32>
      scf.yield %12, %13 : tensor<64xi32>, tensor<64xi32>
    }
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   tt.func public @gather_simple_no_mask
// CHECK-SAME:    ([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK-DAG:       [[CST_10s_:%.+]] = arith.constant dense<10> : tensor<64xi32>
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : i32
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i32
// CHECK-DAG:       [[CST_64s_:%.+]] = arith.constant dense<64> : tensor<64xi32>
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : !tt.ptr<f32> to memref<*xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_0_]] : !tt.ptr<f32> to memref<*xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK:           [[VAR_3_:%.+]]:2 = scf.for [[VAR_4_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_5_:%.+]] = [[VAR_2_]], [[VAR_6_:%.+]] = [[VAR_2_]]) -> (tensor<64xi32>, tensor<64xi32>)  : i32 {
// CHECK:             [[VAR_7_:%.+]] = arith.divsi [[VAR_5_]], [[CST_10s_]] : tensor<64xi32>
// CHECK:             [[VAR_8_:%.+]] = arith.addi [[VAR_4_]], [[CST_5_]] : i32
// CHECK:             [[VAR_9_:%.+]] = arith.remsi [[VAR_8_]], [[CST_64_]] : i32
// CHECK:             [[VAR_10_:%.+]] = tt.splat [[VAR_9_]] : i32 -> tensor<64xi32>
// CHECK:             [[VAR_11_:%.+]] = arith.addi [[VAR_7_]], [[VAR_10_]] : tensor<64xi32>
// CHECK:             [[VAR_12_:%.+]] = memref.cast [[VAR_1_]] : memref<*xf32> to memref<?xf32>
// CHECK:             [[VAR_13_:%.+]] = bufferization.to_tensor [[VAR_12_]] restrict : memref<?xf32> to tensor<?xf32>
// CHECK:             [[VAR_14_:%.+]] = tensor.empty() : tensor<64xf32>
// CHECK:             [[VAR_15_:%.+]] = linalg.generic {indexing_maps = [[[MAP_0_]], [[MAP_0_]]], iterator_types = ["parallel"]} ins([[VAR_11_]] : tensor<64xi32>) outs([[VAR_14_]] : tensor<64xf32>) {
// CHECK:             ^bb0([[IN_0_:%.+]]: i32, [[IN_1_:%.+]]: f32):
// CHECK:               [[VAR_16_:%.+]] = arith.index_cast [[IN_0_]] : i32 to index
// CHECK:               [[VAR_17_:%.+]] = tensor.extract [[VAR_13_]]{{.}}[[VAR_16_]]{{.}} : tensor<?xf32>
// CHECK:               linalg.yield [[VAR_17_]] : f32
// CHECK:             } -> tensor<64xf32>
// CHECK:             [[VAR_18_:%.+]] = memref.cast [[VAR_0_]] : memref<*xf32> to memref<?xf32>
// CHECK:             linalg.generic {indexing_maps = [[[MAP_0_]], [[MAP_0_]]], iterator_types = ["parallel"]} ins([[VAR_6_]], [[VAR_15_]] : tensor<64xi32>, tensor<64xf32>) {
// CHECK:             ^bb0([[IN_2_:%.+]]: i32, [[IN_3_:%.+]]: f32):
// CHECK:               [[VAR_19_:%.+]] = arith.index_cast [[IN_2_]] : i32 to index
// CHECK:               memref.store [[IN_3_]], [[VAR_18_]]{{.}}[[VAR_19_]]{{.}} : memref<?xf32>
// CHECK:               linalg.yield
// CHECK:             }
// CHECK:             [[VAR_20_:%.+]] = arith.addi [[VAR_11_]], [[CST_64s_]] : tensor<64xi32>
// CHECK:             [[VAR_21_:%.+]] = arith.addi [[VAR_6_]], [[CST_64s_]] : tensor<64xi32>
// CHECK:             scf.yield [[VAR_20_]], [[VAR_21_]] : tensor<64xi32>, tensor<64xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }