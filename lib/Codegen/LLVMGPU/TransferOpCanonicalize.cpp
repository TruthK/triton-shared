#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/PatternMatch.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

using namespace mlir;
using namespace mlir::tts;

namespace mlir::tts {
#define GEN_PASS_DEF_TRANSFEROPCANONICALIZEPASS
#include "triton-shared/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// 单一Pattern: 针对bufferization::ToTensorOp，链式匹配ToMemrefOp和TransferOps并重写
class OptimizeBufferizationTransferPattern
    : public OpRewritePattern<bufferization::ToTensorOp> {
public:
  using OpRewritePattern<bufferization::ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::ToTensorOp toTensorOp,
                                PatternRewriter &rewriter) const override {
    Value tensorVal = toTensorOp.getResult();
    // 遍历所有ToMemrefOp
    for (auto &use : tensorVal.getUses()) {
      auto toMemrefOp = dyn_cast<bufferization::ToMemrefOp>(use.getOwner());
      if (!toMemrefOp) continue;
      Value memrefVal = toMemrefOp.getResult();
      // 遍历所有TransferOp
      for (auto &memUse : memrefVal.getUses()) {
        Operation *userOp = memUse.getOwner();
        // vector.transfer_read
        if (auto readOp = dyn_cast<vector::TransferReadOp>(userOp)) {
          Value origMem = toTensorOp.getMemref();
          auto newRead = rewriter.create<vector::TransferReadOp>(
              readOp.getLoc(), readOp.getVectorType(), origMem,
              readOp.getIndices(), readOp.getPermutationMap(),
              readOp.getPadding(), readOp.getMask(), readOp.getInBoundsAttr());
          rewriter.replaceOp(readOp, newRead.getResult());
          return success();
        }
        // IREE::VectorExt::TransferWriteOp
        if (auto writeOp = dyn_cast<IREE::VectorExt::TransferWriteOp>(userOp)) {
          Value origMem = toTensorOp.getMemref();
          auto newWrite = rewriter.create<IREE::VectorExt::TransferWriteOp>(
              writeOp.getLoc(), writeOp.getBase(),origMem, 
              writeOp.getMixedIndices(), writeOp.getMixedMaskDims());
          rewriter.replaceOp(writeOp, newWrite);
          return success();
        }
      }
    }
    return failure();
  }
};

class TransferOpCanonicalizePass
    : public impl::TransferOpCanonicalizePassBase<TransferOpCanonicalizePass> {
public:
  using impl::TransferOpCanonicalizePassBase<
      TransferOpCanonicalizePass>::TransferOpCanonicalizePassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // 添加单一Bufferization Transfer优化模式
    patterns.add<OptimizeBufferizationTransferPattern>(context);
    // 应用模式
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::tts