

#include "triton-shared/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "triton-shared/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "triton-shared/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "triton-shared/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "triton-shared/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "triton-shared/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tts::GPU {

#define GEN_PASS_DEF_UNROLLTOINTRINSICSPASS
#include "triton-shared/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {
struct UnrollToIntrinsicsPass final
    : impl::UnrollToIntrinsicsPassBase<UnrollToIntrinsicsPass> {
  void runOnOperation() override;
};
} // namespace

void UnrollToIntrinsicsPass::runOnOperation() {
  MLIRContext *context = &getContext();

  {
    RewritePatternSet patterns(context);
    GPU::populateIREEGPUVectorUnrollPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  // Post unrolling unit dim folding patterns in preparation for later
  // lowerings.
  {
    RewritePatternSet patterns(context);
    GPU::populateIREEGPUDropUnitDimsPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

} // namespace mlir::tts::GPU
