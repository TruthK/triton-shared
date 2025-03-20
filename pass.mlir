func.func @main$async_dispatch_0_matmul_1024x1024x512_f32() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUMatmulTensorCoreMmaSync workgroup_size = [64, 2, 1] subgroup_size = 32, {pipeline_depth = 4 : i64, store_stage = 1 : i64}>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<1024x512xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x512xf32>> -> tensor<1024x512xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>> -> tensor<512x1024xf32>
  %5 = tensor.empty() : tensor<1024x1024xf32>
  %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>} ins(%cst : f32) outs(%5 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %7 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>} ins(%3, %4 : tensor<1024x512xf32>, tensor<512x1024xf32>) outs(%6 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : tensor<1024x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x1024xf32>>
  return
}