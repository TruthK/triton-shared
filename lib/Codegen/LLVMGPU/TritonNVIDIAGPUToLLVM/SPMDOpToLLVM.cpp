#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton-shared/Codegen/LLVMGPU/TritonNVIDIAGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::tts;

namespace {

struct GetNumProgramsOpConversion
    : public OpRewritePattern<triton::GetNumProgramsOp> {
  using OpRewritePattern<triton::GetNumProgramsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::GetNumProgramsOp op,
                                PatternRewriter &rewriter) const override {
    static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                    mlir::gpu::Dimension::y,
                                                    mlir::gpu::Dimension::z};
    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);
    Value blockId =
        rewriter.create<::mlir::gpu::GridDimOp>(loc, dims[op.getAxisAsInt()]);
    Type i32_ty = rewriter.getIntegerType(32);
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, i32_ty, blockId);
    return success();
  }
};

struct GetProgramIdOpConversion
    : public OpRewritePattern<triton::GetProgramIdOp> {

  using OpRewritePattern<triton::GetProgramIdOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::GetProgramIdOp op,
                                PatternRewriter &rewriter) const override {
    static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                    mlir::gpu::Dimension::y,
                                                    mlir::gpu::Dimension::z};

    Location loc = op->getLoc();
    Type i32_ty = rewriter.getIntegerType(32);
    Value blockId =
        rewriter.create<::mlir::gpu::BlockIdOp>(loc, dims[op.getAxisAsInt()]);

    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, i32_ty, blockId);
    return success();
  }
};

} // namespace

void mlir::tts::NVIDIA::populateTTSSPMDOpToLLVMPattern(
    RewritePatternSet &patterns) {
  patterns.add<GetNumProgramsOpConversion>(patterns.getContext());
  patterns.add<GetProgramIdOpConversion>(patterns.getContext());
}
