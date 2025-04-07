// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "triton-shared/Codegen/Common/Passes.h"
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tts;
using namespace mlir::tts::IREE::VectorExt;

namespace mlir::tts {

#define GEN_PASS_DEF_VECTOREXTTRANSFERTOVECTORTRANSFERPASS
#include "triton-shared/Codegen/Common/Passes.h.inc"

namespace {

/// 将 iree_vector_ext.transfer_read 转换为 vector.transfer_read
struct ConvertVectorExtTransferReadToVectorTransferRead
    : public OpRewritePattern<IREE::VectorExt::TransferReadOp> {
  using OpRewritePattern<IREE::VectorExt::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::VectorExt::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    // 检查TransferReadOp的使用者是否为memref.copy
    bool hasMemrefCopyUser = false;
    for (Operation *user : op->getUsers()) {
      if (isa<memref::CopyOp>(user)) {
        hasMemrefCopyUser = true;
        break;
      }
    }

    // 如果不是memref.copy使用者，则报错
    if (!hasMemrefCopyUser) {
      return op->emitError(
          "iree_vector_ext.transfer_read must be used by memref.copy");
    }

    // 获取源和结果类型
    MemRefType resultType = mlir::cast<MemRefType>(op.getResult().getType());

    // 创建向量类型作为vector.transfer_read的结果类型
    VectorType vectorType =
        VectorType::get(resultType.getShape(), resultType.getElementType());

    // 获取原始的indices和mask_dims
    SmallVector<OpFoldResult> mixedIndices = op.getMixedIndices();
    SmallVector<OpFoldResult> mixedMaskDims = op.getMixedMaskDims();

    // 计算mask：min(mask[dim], indices[dim] + resultShape[dim])
    Location loc = op.getLoc();
    SmallVector<Value> maskValues;
    for (unsigned i = 0; i < mixedMaskDims.size(); ++i) {
      OpFoldResult maskDim = mixedMaskDims[i];
      int64_t resultDimSize = vectorType.getDimSize(i);

      // 获取mask维度的值
      Value maskDimValue;
      if (auto attr = maskDim.dyn_cast<Attribute>()) {
        // 静态尺寸
        int64_t staticSize = mlir::cast<IntegerAttr>(attr).getInt();
        maskDimValue = rewriter.create<arith::ConstantIndexOp>(loc, staticSize);
      } else {
        // 动态尺寸
        maskDimValue = maskDim.dyn_cast<Value>();
      }

      // 获取indices维度的值
      Value indexValue;
      if (i < op.getIndices().size()) {
        indexValue = op.getIndices()[i];
      } else {
        // 如果没有对应的索引，默认为0
        indexValue = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      }

      // 计算 indices[dim] + resultShape[dim]
      Value resultDimSizeValue = rewriter.create<arith::ConstantIndexOp>(loc, resultDimSize);
      Value indicesPlusSize = rewriter.create<arith::AddIOp>(loc, indexValue, resultDimSizeValue);

      // 计算 min(mask[dim], indices[dim] + resultShape[dim])
      Value minVal = rewriter.create<arith::MinSIOp>(loc, maskDimValue, indicesPlusSize);
      maskValues.push_back(minVal);
    }

    // 创建vector.transfer_read操作
    Value vectorResult;
    
    // 创建恒等AffineMap
    auto identityMap = AffineMap::getMultiDimIdentityMap(mixedIndices.size(),
                                                         rewriter.getContext());
    auto permMapAttr = AffineMapAttr::get(identityMap);
    
    // 创建mask使用vector.create_mask
    Value mask = rewriter.create<vector::CreateMaskOp>(
        loc, VectorType::get(vectorType.getShape(), rewriter.getI1Type()), maskValues);
    
    // 获取paddingValue（可能为空）
    Value paddingValue = op.getOther();
    // 如果没有提供paddingValue，创建一个默认的0值
    if (!paddingValue) {
      Type elementType = resultType.getElementType();
      paddingValue = rewriter.create<arith::ConstantOp>(
          loc, elementType, rewriter.getZeroAttr(elementType));
    }
    
    // 提前创建inBounds属性
    SmallVector<bool> inBoundsValues(mixedIndices.size(), true);
    auto inBoundsAttr = rewriter.getBoolArrayAttr(inBoundsValues);
    
    // 使用正确的参数列表创建vector.transfer_read
    auto vecReadOp = rewriter.create<vector::TransferReadOp>(
        loc,
        vectorType,
        op.getBase(),
        ValueRange(op.getIndices()),
        permMapAttr,
        paddingValue,
        mask,
        inBoundsAttr);
    
    vectorResult = vecReadOp.getResult();
    
    // 收集需要处理的memref.copy操作
    SmallVector<memref::CopyOp> copyOps;
    for (Operation *user : op->getUsers()) {
      if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
        copyOps.push_back(copyOp);
      }
    }
    
    // 替换所有memref.copy的使用
    for (auto copyOp : copyOps) {
      Value dest = copyOp.getTarget();
      
      // 创建所有索引为0的ValueRange
      SmallVector<Value> zeroIndices;
      for (int i = 0; i < vectorType.getRank(); ++i) {
        zeroIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }
      
      // 创建vector.store替代memref.copy和vector.transfer_write
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(copyOp);
      auto vectorStoreOp = rewriter.create<vector::StoreOp>(
          copyOp.getLoc(), vectorResult, dest, ValueRange(zeroIndices));
      
      // 删除原始的memref.copy操作，但保持原有的操作顺序
      rewriter.replaceOp(copyOp, vectorStoreOp->getResults());
    }
    
    // 删除原始的iree_vector_ext.transfer_read操作
    rewriter.replaceOp(op, vectorResult);

    return success();
  }
};

struct VectorExtTransferToVectorTransferPass
    : impl::VectorExtTransferToVectorTransferPassBase<
          VectorExtTransferToVectorTransferPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // 添加转换模式
    patterns.add<ConvertVectorExtTransferReadToVectorTransferRead>(context);

    // 应用模式
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

} // namespace mlir::tts