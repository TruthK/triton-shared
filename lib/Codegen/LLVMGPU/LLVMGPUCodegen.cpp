#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include "triton-shared/Codegen/Common/GPU/Passes.h"
#include "triton-shared/Codegen/Common/PassUtils.h"
#include "triton-shared/Codegen/Common/Passes.h"
#include "triton-shared/Codegen/LLVMGPU/Passes.h"

namespace mlir::tts {

#define GEN_PASS_DEF_LLVMGPUCODEGENPASS
#include "triton-shared/Codegen/LLVMGPU/Passes.h.inc"

namespace {

// Add passes to make the address computation more explicit and optimize them.
//
// The idea here is to be less dependent on what the LLVM backend is able to do,
// by heavy lifting most of the work while we still have the information about
// loops.
//
// Note that this needs to run before SCF -> CF.
static void
addLowerAndOptimizeAddressComputationPasses(FunctionLikeNest &funcPassManager) {
  funcPassManager.addPass(createExtractAddressComputationGPUPass)
      .addPass(memref::createExpandOpsPass)
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass(memref::createExpandStridedMetadataPass)
      // Hoist loop invariant variables to give affine decomposition pass the
      // right loop dependencies.
      .addPass(createIREELoopInvariantCodeMotionPass)
      // Decompose affine ops.
      .addPass(createDecomposeAffineOpsPass)
      // Get rid of the redundant computations.
      .addPass(createCSEPass)
      // Hoist the resulting decompositions.
      .addPass(createIREELoopInvariantCodeMotionPass)
      .addPass(affine::createAffineExpandIndexOpsPass)
      .addPass(createLowerAffinePass)
      // Do another round of LICM now that we've lowered and optimized
      // arithmetic
      .addPass(createCSEPass)
      .addPass(createIREELoopInvariantCodeMotionPass);
}

static void addLowerToLLVMGPUPasses(OpPassManager &modulePassManager) {
  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());

  // modulePassManager.addPass(createLowerUKernelOpsToCallsPass());
  FunctionLikeNest(modulePassManager)
      // Linalg -> SCF
      .addPass(createMemrefCopyToLinalgPass)
      .addPass(createConvertLinalgToLoopsPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      // Pad allocations with dynamic dimension after linalg lowering but before
      // lowering SCF and affine ops.
      // .addPass(createPadDynamicAllocPass)
      // Hoist any newly static allocations from PadDynamicAlloc.
      .addPass(createHoistStaticallyBoundAllocationsPass)
      .addPass(createLowerAffinePass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);

  // Handled tensor constants.
  addConstantBufferizePasses(modulePassManager);

  FunctionLikeNest funcPassManager(modulePassManager);
  funcPassManager.addPass(createFoldTensorExtractOpPass)
      .addPass(createLLVMGPUVectorLoweringPass)
      .addPass(createExpandGPUOpsPass);
  // Expose workitem and workgroup counts to range inference later.
  // .addPass(createGPUPropagateDispatchSizeBoundsPass);

  // This pass needs to run before SCF -> CF.
  addLowerAndOptimizeAddressComputationPasses(funcPassManager);

  // Run checks on shared memory usage.
  funcPassManager
      .addPass([&]() {
        auto getIndexBitwidth = [](mlir::FunctionOpInterface) { return 64; };
        return createGPUCheckResourceUsagePass(getIndexBitwidth);
      })
      // SCF -> CF
      .addPass(createConvertSCFToCFPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      // Handle complex operation conversion.
      .addPass(createConvertComplexToStandardPass)
      // Convert BF16 operations to occur as F32.
      .addPass(createConvertBf16ArithToF32Pass)
      .addPass(createConvertBf16ToUInt16BuffersPass)
      // Convert math dialect elementry functions to polynomial form.
      .addPass(createPolynomialApproximationPass)
      .addPass(memref::createExpandOpsPass)
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass(memref::createExpandStridedMetadataPass)
      .addPass(createEmulateNarrowTypePass)
      .addPass(affine::createAffineExpandIndexOpsPass)
      .addPass(createLowerAffinePass);

  // Strip out the debug info for the kernel.
  modulePassManager.addPass(createStripDebugInfoPass());
  // Cast address spaces of all function arguments to generic.
  modulePassManager.addPass(createLLVMGPUCastAddressSpaceFunctionPass());
  // convert to NVVM.
  modulePassManager.addPass(createConvertToNVVMPass());
}

class LLVMGPUCodegenPass final
    : public impl::LLVMGPUCodegenPassBase<LLVMGPUCodegenPass> {
public:
  using impl::LLVMGPUCodegenPassBase<
      LLVMGPUCodegenPass>::LLVMGPUCodegenPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<affine::AffineDialect,
                arith::ArithDialect,
                bufferization::BufferizationDialect,
                func::FuncDialect,
                gpu::GPUDialect,
                linalg::LinalgDialect,
                memref::MemRefDialect,
                scf::SCFDialect,
                tensor::TensorDialect,
                vector::VectorDialect,
                nvgpu::NVGPUDialect,
                NVVM::NVVMDialect>();
    // clang-format on
  }

  void runOnOperation() override {
    // 获取当前 Module
    ModuleOp moduleOp = getOperation();

    PassManager pm(&getContext(), moduleOp.getOperationName());
    // 创建一个新的 OpPassManager 针对 ModuleOp
    FunctionLikeNest(pm)
        .addPass(createLLVMGPULowerExecutableTargetPass)
        .addPass(createVerifyWorkgroupDistributionPass);

    pm.addPass(createReconcileTranslationInfoPass());

    //===--------------------------------------------------------------------===//
    // Convert Linalg ops to LLVM+NVVM/ROCDL ops.
    //
    // Post-conditions:
    //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
    //   - The module contains the final llvm.module ready to be serialized.
    //===--------------------------------------------------------------------===//
    addLowerToLLVMGPUPasses(pm);
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
      llvm::dbgs() << " Using LLVMGPU pass pipeline: G! \n";
    }
  }
};

} // namespace

} // namespace mlir::tts