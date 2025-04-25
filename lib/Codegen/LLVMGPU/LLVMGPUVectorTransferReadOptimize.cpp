#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

using namespace mlir;

namespace mlir::tts {
#define GEN_PASS_DEF_LLVMGPUVECTORTRANSFERREADOPTIMIZEPASS
#include "triton-shared/Codegen/LLVMGPU/Passes.h.inc"

namespace {

// 为 vector.transfer_read 定义优化模式
class OptimizeVectorTransferRead
    : public OpRewritePattern<vector::TransferReadOp> {
public:
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp transferOp,
                                PatternRewriter &rewriter) const override {
    // 检查 vector.transfer_read 的源是否来自 bufferization.to_memref
    auto memRefSource = transferOp.getSource();
    auto toMemrefOp = memRefSource.getDefiningOp<bufferization::ToMemrefOp>();
    if (!toMemrefOp)
      return failure();

    // 检查 bufferization.to_memref 的源是否来自 bufferization.to_tensor
    auto tensorSource = toMemrefOp.getTensor();
    auto toTensorOp = tensorSource.getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensorOp)
      return failure();

    // 优化: 直接使用原始 memref 作为 vector.transfer_read 的源
    auto originalMemref = toTensorOp.getMemref();

    // 创建新的 vector.transfer_read
    auto newTransferOp = rewriter.create<vector::TransferReadOp>(
        transferOp.getLoc(), transferOp.getVectorType(), originalMemref,
        transferOp.getIndices(), transferOp.getPermutationMap(),
        transferOp.getPadding(), transferOp.getMask(),
        transferOp.getInBoundsAttr());

    rewriter.replaceOp(transferOp, newTransferOp.getResult());
    return success();
  }
};

class LLVMGPUVectorTransferReadOptimizePass
    : public impl::LLVMGPUVectorTransferReadOptimizePassBase<
          LLVMGPUVectorTransferReadOptimizePass> {
public:
  using impl::LLVMGPUVectorTransferReadOptimizePassBase<
      LLVMGPUVectorTransferReadOptimizePass>::LLVMGPUVectorTransferReadOptimizePassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // 添加优化模式
    patterns.add<OptimizeVectorTransferRead>(context);

    // 应用模式
    if (failed(applyPatternsGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::tts