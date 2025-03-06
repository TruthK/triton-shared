

//===----------------------------------------------------------------------===//
// This file implements the pass to combine multiple `tts_gpu.barrier_region`
// ops.
//===----------------------------------------------------------------------===//

#include "triton-shared/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "triton-shared/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "llvm/Support/Casting.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "tts-gpu-combine-barrier-regions"

namespace mlir::tts::GPU {

#define GEN_PASS_DEF_COMBINEBARRIERREGIONSPASS
#include "triton-shared/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {

struct CombineBarrierRegionsPass final
    : impl::CombineBarrierRegionsPassBase<CombineBarrierRegionsPass> {
  void runOnOperation() override;
};

/// Given two barriers, barrierA and barrierB where A is the op immediately
/// before B in the block, combine them into a single barrier.
static LogicalResult
combineBarrierRegionPair(RewriterBase &rewriter,
                         tts::GPU::BarrierRegionOp barrierA,
                         tts::GPU::BarrierRegionOp barrierB) {
  assert(barrierA->getBlock() == barrierB->getBlock() &&
         barrierA->getNextNode() == barrierB && "Expected adjacent barriers");

  // Fail if barrierA is used by barrierB, either directly or by implicit
  // capture.
  for (auto user : barrierA->getUsers()) {
    if (user == barrierB || barrierB->isProperAncestor(user)) {
      return failure();
    }
  }

  Location fusedLoc =
      rewriter.getFusedLoc({barrierA.getLoc(), barrierB.getLoc()});

  // Get the combined operands, result types, and yielded values.
  SmallVector<Value> combinedOperands = barrierA.getInputs();
  combinedOperands.append(barrierB.getInputs().begin(),
                          barrierB.getInputs().end());
  SmallVector<Type> combinedTypes(barrierA->getResultTypes());
  combinedTypes.append(barrierB->getResultTypes().begin(),
                       barrierB->getResultTypes().end());

  auto aYield = cast<tts::GPU::YieldOp>(barrierA.getBody()->getTerminator());
  auto bYield = cast<tts::GPU::YieldOp>(barrierB.getBody()->getTerminator());
  SmallVector<Value> combinedYields = aYield.getValues();
  combinedYields.append(bYield.getValues().begin(), bYield.getValues().end());

  // Create the new barrier op.
  auto combinedBarrierOp = rewriter.create<tts::GPU::BarrierRegionOp>(
      fusedLoc, combinedTypes, combinedOperands);

  MutableArrayRef<BlockArgument> barrierABbArgReplacements =
      combinedBarrierOp.getBody()->getArguments().take_front(
          barrierA->getNumOperands());
  MutableArrayRef<BlockArgument> barrierBBbArgReplacements =
      combinedBarrierOp.getBody()->getArguments().take_back(
          barrierB->getNumOperands());

  // Merge the bodies of the old barriers into the new one.
  rewriter.mergeBlocks(barrierA.getBody(), combinedBarrierOp.getBody(),
                       barrierABbArgReplacements);
  rewriter.mergeBlocks(barrierB.getBody(), combinedBarrierOp.getBody(),
                       barrierBBbArgReplacements);

  // Erase the old terminators and create a new one with a concatenated list of
  // values.
  rewriter.eraseOp(aYield);
  rewriter.eraseOp(bYield);

  rewriter.setInsertionPointToEnd(combinedBarrierOp.getBody());
  rewriter.create<tts::GPU::YieldOp>(fusedLoc, combinedYields);

  SmallVector<Value> valuesToReplace = barrierA.getResults();
  ValueRange bResults = barrierB.getResults();
  valuesToReplace.append(bResults.begin(), bResults.end());
  rewriter.replaceAllUsesWith(valuesToReplace, combinedBarrierOp.getResults());
  rewriter.eraseOp(barrierA);
  rewriter.eraseOp(barrierB);
  return success();
}

struct CombineAdjacentBarrierRegions final
    : OpRewritePattern<tts::GPU::BarrierRegionOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tts::GPU::BarrierRegionOp barrierOp,
                                PatternRewriter &rewriter) const override {
    auto prevBarrier = llvm::dyn_cast_if_present<tts::GPU::BarrierRegionOp>(
        barrierOp->getPrevNode());
    if (!prevBarrier) {
      return failure();
    }
    return combineBarrierRegionPair(rewriter, prevBarrier, barrierOp);
  }
};

void CombineBarrierRegionsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  // These two patterns are run to a fixed point, allowing fusion within
  // potentially nested loops, hoisting from said loops, and continued fusion.
  patterns.add<CombineAdjacentBarrierRegions>(context);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }

  return;
}

} // namespace

} // namespace mlir::tts::GPU
