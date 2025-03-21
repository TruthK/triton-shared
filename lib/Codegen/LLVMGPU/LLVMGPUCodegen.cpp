#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "triton-shared/Codegen/Common/PassUtils.h"
#include "triton-shared/Codegen/LLVMGPU/Passes.h"

namespace mlir::tts {

#define GEN_PASS_DEF_LLVMGPUCODEGENPASS
#include "triton-shared/Codegen/LLVMGPU/Passes.h.inc"

namespace {
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
                vector::VectorDialect>();
    // clang-format on
  }

  void runOnOperation() override {
    // 获取当前 Module
    ModuleOp moduleOp = getOperation();

    PassManager pm(&getContext(), moduleOp.getOperationName());
    // 创建一个新的 OpPassManager 针对 ModuleOp
    FunctionLikeNest(pm).addPass(createLLVMGPULowerExecutableTargetPass);
    // .addPass(createVerifyWorkgroupDistributionPass);

    //   variantPassManager.addPass(createReconcileTranslationInfoPass());

    //   //===--------------------------------------------------------------------===//
    //   // Convert Linalg ops to LLVM+NVVM/ROCDL ops.
    //   //
    //   // Post-conditions:
    //   //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
    //   //   - The module contains the final llvm.module ready to be
    //   serialized.
    //   //===--------------------------------------------------------------------===//
    //   addLowerToLLVMGPUPasses(variantPassManager.nest<ModuleOp>(), useROCM);

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
      llvm::dbgs() << " Using LLVMGPU pass pipeline: G! \n";
    }
  }
};

} // namespace

} // namespace mlir::tts