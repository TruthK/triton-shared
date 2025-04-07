// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "triton-shared/Codegen/Common/Passes.h"
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::tts;
using namespace mlir::tts::IREE::VectorExt;
namespace mlir::tts {
#define GEN_PASS_DEF_TRANSFERREADSUBVIEWFUSIONPASS
#include "triton-shared/Codegen/Common/Passes.h.inc"

namespace {

/// 将 memref.subview + iree_vector_ext.transfer_read 融合为单个
/// iree_vector_ext.transfer_read 操作
struct FuseSubviewWithTransferRead
    : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp subviewOp,
                                PatternRewriter &rewriter) const override {
    // 只处理 memref.subview 操作
    Value source = subviewOp.getSource();

    // 检查源操作是否为 TransferReadOp
    auto transferOp = source.getDefiningOp<IREE::VectorExt::TransferReadOp>();
    if (!transferOp)
      return failure();

    Location loc = subviewOp->getLoc();

    // 获取 TransferReadOp 的信息
    Value transferBase = transferOp.getBase();
    auto transferIndices = transferOp.getMixedIndices();
    auto transferMaskDims = transferOp.getMixedMaskDims();
    Value other = transferOp.getOther();

    // 获取 SubViewOp 的信息
    SmallVector<OpFoldResult> offsets = subviewOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = subviewOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = subviewOp.getMixedStrides();

    // 计算新的索引，基于 SubViewOp 的偏移和 TransferOp 的索引
    SmallVector<OpFoldResult> newIndices;
    for (unsigned i = 0; i < offsets.size(); ++i) {
      // 如果索引超出范围，使用原始 transferIndices
      if (i >= transferIndices.size()) {
        if (i < offsets.size())
          newIndices.push_back(offsets[i]);
        continue;
      }

      // 合并偏移量和原始索引
      Value combinedIndex;
      Value indexValue;

      // 获取原始索引
      if (auto attr = transferIndices[i].dyn_cast<Attribute>()) {
        int64_t staticIndex = cast<IntegerAttr>(attr).getInt();
        indexValue = rewriter.create<arith::ConstantIndexOp>(loc, staticIndex);
      } else {
        indexValue = cast<Value>(transferIndices[i]);
      }

      // 获取偏移量
      if (auto attr = offsets[i].dyn_cast<Attribute>()) {
        int64_t offsetValue = cast<IntegerAttr>(attr).getInt();
        Value offsetIndex =
            rewriter.create<arith::ConstantIndexOp>(loc, offsetValue);
        combinedIndex =
            rewriter.create<arith::AddIOp>(loc, indexValue, offsetIndex);
      } else {
        Value offsetValue = cast<Value>(offsets[i]);
        combinedIndex =
            rewriter.create<arith::AddIOp>(loc, indexValue, offsetValue);
      }

      newIndices.push_back(combinedIndex);
    }

    // 创建新的 transferRead 操作，保持原始 TransferReadOp 的 mask
    auto result = rewriter.create<mlir::tts::IREE::VectorExt::TransferReadOp>(
        loc, subviewOp.getType(), transferBase, newIndices, transferMaskDims, other);

    // 替换 SubViewOp 为新的 TransferReadOp
    rewriter.replaceOp(subviewOp, result);

    return success();
  }
};

struct TransferReadSubviewFusionPass
    : impl::TransferReadSubviewFusionPassBase<TransferReadSubviewFusionPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // 添加融合模式
    patterns.add<FuseSubviewWithTransferRead>(context);

    // 应用模式
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // 执行 Canonicalizer 和 CSE
    OpPassManager dynamicPM(getOperation()->getName());
    dynamicPM.addPass(mlir::createCanonicalizerPass());
    dynamicPM.addPass(mlir::createCSEPass());

    if (failed(runPipeline(dynamicPM, getOperation()))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

} // namespace mlir::tts