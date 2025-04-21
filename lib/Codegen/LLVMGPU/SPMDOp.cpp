#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "triton-shared/Codegen/LLVMGPU/Passes.h"
#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace mlir::tts {

#define GEN_PASS_DEF_SPMDOPPASS
#include "triton-shared/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// Selects a lowering strategy for taking a hal.executable.variant operation
/// to scalar/native-vector code.
class SPMDOpPass final : public impl::SPMDOpPassBase<SPMDOpPass> {
public:
  using impl::SPMDOpPassBase<SPMDOpPass>::SPMDOpPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect, mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void SPMDOpPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ConversionTarget target(*context);

  target.addLegalOp<arith::IndexCastOp, mlir::gpu::BlockIdOp, arith::TruncIOp,
                    mlir::gpu::GridDimOp>();
  target.addIllegalOp<triton::GetNumProgramsOp, triton::GetProgramIdOp>();
  target.addIllegalDialect<triton::TritonDialect>();
  // 添加转换模式
  RewritePatternSet patterns(context);
  mlir::tts::NVIDIA::populateTTSSPMDOpToLLVMPattern(patterns);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
} // namespace mlir::tts
