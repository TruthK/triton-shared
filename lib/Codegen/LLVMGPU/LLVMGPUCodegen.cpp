#include "mlir/Dialect/Func/IR/FuncOps.h"
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
      llvm::dbgs() << "ji Using LLVMGPU pass pipeline:\n";
    }
  }
};

} // namespace

} // namespace mlir::tts