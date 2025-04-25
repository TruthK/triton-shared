#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Codegen/LLVMGPU/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

using namespace mlir;

namespace mlir::tts {

#define GEN_PASS_DEF_TENSORTRANSFERWRITEFUSIONPASS
#include "triton-shared/Codegen/LLVMGPU/Passes.h.inc"

namespace {

// Pattern to fuse tensor.parallel_insert_slice with tts.transfer_write
struct FuseTensorTransferWrite : public OpRewritePattern<TransferWriteOp> {
  using OpRewritePattern<TransferWriteOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransferWriteOp transferWriteOp,
                                PatternRewriter &rewriter) const override {
    // Match when value comes from scf.forall
    auto forOp = transferWriteOp.getValue().getDefiningOp<scf::ForallOp>();
    if (!forOp)
      return failure();
    // Only one shared_out / one result
    if (forOp.getResults().size() != 1 || forOp.getOutputs().size() != 1)
      return failure();
    // Initial output must be tensor.empty()
    if (!forOp.getOutputs()[0].getDefiningOp<tensor::EmptyOp>())
      return failure();
    // Find the parallel_insert_slice inside the forall.in_parallel region
    auto inParallelOp = forOp.getTerminator();
    if (!inParallelOp)
      return failure();
    forOp.getOutputs()[0].dump();
    inParallelOp.dump();
    forOp.getRegionIterArgs()[0].dump();
    // Collect matching slice ops
    SmallVector<tensor::ParallelInsertSliceOp> sliceOps;
    Region &innerRegion = inParallelOp->getRegion(0);
    for (Block &block : innerRegion) {
      for (Operation &op : block) {
        if (auto sliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(op)) {
          op.dump();
          if (sliceOp.getDest() == forOp.getRegionIterArgs()[0]) {
            sliceOp.dump();
            sliceOps.push_back(sliceOp);
          }
        }
      }
    }
    if (sliceOps.size() != 1)
      return failure();
    auto sliceOp = sliceOps[0];
    // Compute new indices: origIndices + sliceOffsets
    auto origIdx = transferWriteOp.getMixedIndices();
    auto sliceOff = sliceOp.getMixedOffsets();
    if (origIdx.size() != sliceOff.size())
      return failure();
    SmallVector<OpFoldResult> newIdx;
    Location loc = transferWriteOp.getLoc();
    for (unsigned i = 0, e = origIdx.size(); i < e; ++i)
      newIdx.push_back(addOFRs(origIdx[i], sliceOff[i], loc, rewriter));
    // Create a new tts.transfer_write inside the parallel block
    rewriter.setInsertionPoint(sliceOp);
    rewriter.create<TransferWriteOp>(
        sliceOp.getLoc(), transferWriteOp.getBase(), sliceOp.getSource(),
        newIdx, transferWriteOp.getMixedMaskDims());
    // Erase the original slice and the outer transfer_write
    rewriter.eraseOp(sliceOp);
    rewriter.eraseOp(transferWriteOp);
    return success();
  }
};

// 定义Pass
struct TensorTransferWriteFusionPass
    : public impl::TensorTransferWriteFusionPassBase<
          TensorTransferWriteFusionPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FuseTensorTransferWrite>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tts