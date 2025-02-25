
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton-shared/Conversion/LinalgToLLVM/Passes.h"

namespace mlir::tts {

#define GEN_PASS_DEF_MEMREFCOPYTOLINALGPASS
#include "triton-shared/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {
/// Create a linalg::GenericOp version of an n-D copy that can further tile,
/// lower to loops or vectorize, unlike the current implementation of
/// memref::CopyOp.
Operation *createLinalgCopyOp(OpBuilder &b, Location loc, Value from, Value to,
                              ArrayRef<NamedAttribute> attributes) {
  auto memrefTypeFrom = llvm::dyn_cast<MemRefType>(from.getType());
  auto memrefTypeTo = llvm::dyn_cast<MemRefType>(to.getType());
  if (!memrefTypeFrom || !memrefTypeTo ||
      memrefTypeFrom.getRank() != memrefTypeTo.getRank()) {
    mlir::emitError(
        loc, "unable to generate copy op within bufferization from type ")
        << memrefTypeFrom << " to " << memrefTypeTo;
    return nullptr;
  }
  AffineMap id =
      AffineMap::getMultiDimIdentityMap(memrefTypeTo.getRank(), b.getContext());
  SmallVector<utils::IteratorType> iteratorTypes(memrefTypeTo.getRank(),
                                                 utils::IteratorType::parallel);
  return b.create<linalg::GenericOp>(
      loc,
      /*inputs=*/from,
      /*outputs=*/to,
      /*indexingMaps=*/llvm::ArrayRef({id, id}),
      /*iteratorTypes=*/iteratorTypes,
      [](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args.front());
      },
      attributes);
}

namespace {
struct MemrefCopyOpToLinalg : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Operation *linalgCopy =
        createLinalgCopyOp(rewriter, copyOp.getLoc(), copyOp.getSource(),
                           copyOp.getTarget(), copyOp->getAttrs());
    if (!linalgCopy)
      return failure();
    rewriter.replaceOp(copyOp, linalgCopy->getResults());
    return success();
  }
};
} // namespace
struct MemrefCopyToLinalgPass final
    : impl::MemrefCopyToLinalgPassBase<MemrefCopyToLinalgPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<MemrefCopyOpToLinalg>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tts
