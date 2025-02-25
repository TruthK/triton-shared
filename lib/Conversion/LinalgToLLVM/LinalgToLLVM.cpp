#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Conversion/LinalgToLLVM/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Utils/PassUtils.h"

namespace mlir::tts {

#define GEN_PASS_DEF_LINALGTOLLVMPASS
#include "triton-shared/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

using FunctionLikeNest = MultiOpNest<func::FuncOp>;

static void addCleanupPatterns(OpPassManager &passManager) {
  FunctionLikeNest(passManager)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);
  passManager.addPass(mlir::createSymbolDCEPass());
}

//===----------------------------------------------------------------------===//
// Common pipeline
//===----------------------------------------------------------------------===//
static void
buildCommonPassPipeline(OpPassManager &pm,
                        const mlir::gpu::GPUToNVVMPipelineOptions &options) {
  pm.addPass(createConvertNVGPUToNVVMPass());
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createConvertVectorToSCFPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertNVVMToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());

  GpuNVVMAttachTargetOptions nvvmTargetOptions;
  nvvmTargetOptions.triple = options.cubinTriple;
  nvvmTargetOptions.chip = options.cubinChip;
  nvvmTargetOptions.features = options.cubinFeatures;
  nvvmTargetOptions.optLevel = options.optLevel;
  pm.addPass(createGpuNVVMAttachTarget(nvvmTargetOptions));
  pm.addPass(createLowerAffinePass());
  pm.addPass(createArithToLLVMConversionPass());
  ConvertIndexToLLVMPassOptions convertIndexToLLVMPassOpt;
  convertIndexToLLVMPassOpt.indexBitwidth = options.indexBitWidth;
  pm.addPass(createConvertIndexToLLVMPass(convertIndexToLLVMPassOpt));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

//===----------------------------------------------------------------------===//
// GPUModule-specific stuff.
//===----------------------------------------------------------------------===//
static void
buildGpuPassPipeline(OpPassManager &pm,
                     const mlir::gpu::GPUToNVVMPipelineOptions &options) {
  pm.addNestedPass<gpu::GPUModuleOp>(createStripDebugInfoPass());
  ConvertGpuOpsToNVVMOpsOptions opt;
  opt.useBarePtrCallConv = options.kernelUseBarePtrCallConv;
  opt.indexBitwidth = options.indexBitWidth;
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToNVVMOps(opt));
  pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());
}


class LinalgToLLVMPass : public impl::LinalgToLLVMPassBase<LinalgToLLVMPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        mlir::ttx::TritonTilingExtDialect, mlir::tts::TritonStructuredDialect,
        mlir::triton::TritonDialect, affine::AffineDialect, arith::ArithDialect,
        bufferization::BufferizationDialect, cf::ControlFlowDialect,
        complex::ComplexDialect, func::FuncDialect, gpu::GPUDialect,
        index::IndexDialect, linalg::LinalgDialect, LLVM::LLVMDialect,
        math::MathDialect, memref::MemRefDialect, nvgpu::NVGPUDialect,
        NVVM::NVVMDialect, scf::SCFDialect, tensor::TensorDialect,
        ub::UBDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    auto moduleOp = getOperation();
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createConvertLinalgToAffineLoopsPass());
    pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());

    bufferization::OneShotBufferizationOptions bufferoptions;
    bufferoptions.allowReturnAllocsFromLoops = true;
    pm.addPass(bufferization::createOneShotBufferizePass(bufferoptions));
    addCleanupPatterns(pm);

    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createLowerAffinePass());
    // Expand complicated MemRef operations before lowering them.
    pm.addPass(memref::createExpandStridedMetadataPass());

    pm.addPass(createConvertVectorToGPUPass(true));
    pm.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
    pm.addPass(createParallelLoopToGpuPass());
    addCleanupPatterns(pm);

    gpu::GPUToNVVMPipelineOptions nvvmOptions;
    nvvmOptions.cubinChip = "sm_89";
    nvvmOptions.cubinFeatures = "+ptx83";
    nvvmOptions.optLevel = 3;
    nvvmOptions.cubinFormat ="binary";
    buildCommonPassPipeline(pm, nvvmOptions);
    buildGpuPassPipeline(pm, nvvmOptions);

    addCleanupPatterns(pm);
    pm.addPass(createConvertSCFToCFPass());
    // Convert Func to LLVM (always needed).
    pm.addPass(createConvertFuncToLLVMPass());
    // Expand complicated MemRef operations before lowering them.
    pm.addPass(memref::createExpandStridedMetadataPass());

    pm.addPass(createConvertVectorToLLVMPass());
    pm.addNestedPass<func::FuncOp>(createArithToLLVMConversionPass());
    pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
    pm.addNestedPass<func::FuncOp>(createConvertComplexToLLVMPass());
    addCleanupPatterns(pm);

    pm.addPass(createConvertIndexToLLVMPass());
    pm.addPass(memref::createExpandOpsPass());
    pm.addNestedPass<func::FuncOp>(tts::createMemrefCopyToLinalgPass());
    // Convert MemRef to LLVM (always needed).
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());

    pm.addPass(createLowerAffinePass());

    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    // Convert remaining unrealized_casts (always needed).
    pm.addPass(createReconcileUnrealizedCastsPass());
    addCleanupPatterns(pm);

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tts