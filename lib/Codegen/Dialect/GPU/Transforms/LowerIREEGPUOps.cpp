

#include "triton-shared/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "triton-shared/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "triton-shared/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tts::GPU {

#define GEN_PASS_DEF_LOWERIREEGPUOPSPASS
#include "triton-shared/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {
struct LowerIREEGPUOpsPass final
    : impl::LowerIREEGPUOpsPassBase<LowerIREEGPUOpsPass> {
  void runOnOperation() override;
};
} // namespace

void LowerIREEGPUOpsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  populateIREEGPULowerValueBarrierPatterns(patterns);
  populateIREEGPULowerMultiMmaPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace mlir::tts::GPU
