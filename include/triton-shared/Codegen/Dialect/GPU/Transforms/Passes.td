

#ifndef IREE_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES
#define IREE_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def CombineBarrierRegionsPass :
    Pass<"tts-gpu-combine-barrier-regions", ""> {
  let summary = "Combines `tts_gpu.barrier_region` ops";
  let dependentDialects = ["::mlir::tts::GPU::IREEGPUDialect"];
}

def ConcretizeMmaShapesPass :
    InterfacePass<"tts-gpu-concretize-mma-shapes", "mlir::FunctionOpInterface"> {
  let summary = "Expands the inner dimensions of tts_gpu.multi_mma ops to match the thread layout";
  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::tts::GPU::IREEGPUDialect",
  ];
  let options = [
    Option<"concretizeInputs", "concretize-inputs",
      "bool", /*default=*/"true",
      "Expand the inner dimensions for the lhs and rhs operands of the multi_mma ops.">,
    Option<"concretizeResult", "concretize-result",
      "bool", /*default=*/"true",
      "Expand the inner dimensions for the acc operand of the multi_mma ops.">,
  ];
}

def DistributeMmaToLanesPass :
    InterfacePass<"tts-gpu-distribute-mma-to-lanes", "mlir::FunctionOpInterface"> {
  let summary = "Distributes tts_gpu.multi_mma ops to lanes";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

def LowerIREEGPUOpsPass :
    InterfacePass<"tts-gpu-lower-ops", "mlir::FunctionOpInterface"> {
  let summary = "Post bufferization lowerings of tts_gpu ops before late lowerings";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
  ];
}

def UnrollToIntrinsicsPass :
    InterfacePass<"tts-gpu-unroll-to-intrinsics", "mlir::FunctionOpInterface"> {
  let summary = "Unrolls tts_gpu.multi_mma ops to their inner vector size.";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::vector::VectorDialect",
  ];
}

def VectorizeIREEGPUOpsPass :
    InterfacePass<"tts-gpu-vectorize-ops", "mlir::FunctionOpInterface"> {
  let summary = "Vectorizes then lowers a few tts_gpu ops before vectorization.";
  let dependentDialects = [
    "::mlir::vector::VectorDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::tts::GPU::IREEGPUDialect"
  ];
}

#endif // IREE_CODEGEN_DIALECt_GPU_TRANSFORMS_PASSES
