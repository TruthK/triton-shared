// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "triton-shared/Codegen/Common/Passes.h"
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
    rewriter.setInsertionPoint(op);
    // 获取源和结果类型
    MemRefType resultType = mlir::cast<MemRefType>(op.getResult().getType());
    op.dump();
    resultType.dump();
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
      Value resultDimSizeValue =
          rewriter.create<arith::ConstantIndexOp>(loc, resultDimSize);
      Value indicesPlusSize =
          rewriter.create<arith::AddIOp>(loc, indexValue, resultDimSizeValue);

      // 计算 min(mask[dim], indices[dim] + resultShape[dim])
      Value minVal =
          rewriter.create<arith::MinSIOp>(loc, maskDimValue, indicesPlusSize);

      Value remVal =
          rewriter.create<arith::RemSIOp>(loc, minVal, resultDimSizeValue);
      maskValues.push_back(remVal);
    }

    // 创建恒等AffineMap
    auto identityMap = AffineMap::getMultiDimIdentityMap(mixedIndices.size(),
                                                         rewriter.getContext());
    auto permMapAttr = AffineMapAttr::get(identityMap);

    // 创建mask使用vector.create_mask
    Value mask = rewriter.create<vector::CreateMaskOp>(
        loc, VectorType::get(vectorType.getShape(), rewriter.getI1Type()),
        maskValues);

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

    SmallVector<Value> indices;
    int dimIdx = 0;
    // 遍历所有维度
    for (int i = 0; i < resultType.getRank(); ++i) {
      if (dimIdx < op.getMixedIndices().size()) {
        auto indie = op.getMixedIndices()[dimIdx];
        if (auto attr = indie.dyn_cast<Attribute>()) {
          if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
            indices.push_back(rewriter.create<arith::ConstantIndexOp>(
                op.getLoc(), intAttr.getInt()));
          }
        } else {
          indices.push_back(cast<Value>(indie));
        }
        dimIdx++;
      } else {
        indices.push_back(
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0));
      }
    }
    // 使用正确的参数列表创建vector.transfer_read
    auto vecReadOp = rewriter.create<vector::TransferReadOp>(
        loc, vectorType, op.getBase(), indices, permMapAttr, paddingValue, mask,
        inBoundsAttr);

    auto addressSpace = gpu::AddressSpaceAttr::get(
        rewriter.getContext(), gpu::GPUDialect::getPrivateAddressSpace());

    auto allocType =
        MemRefType::get(resultType.getShape(), resultType.getElementType(),
                        AffineMap(), addressSpace);
    // 创建alloc操作
    auto allocOp = rewriter.create<memref::AllocOp>(loc, allocType);

    // 创建全零索引用于vector.store
    SmallVector<Value> zeroIndices;
    for (int i = 0; i < resultType.getRank(); ++i) {
      zeroIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // 创建vector.store操作
    auto storeOp = rewriter.create<vector::StoreOp>(loc, vecReadOp.getResult(),
                                                    allocOp.getResult(),
                                                    ValueRange(zeroIndices));

    // 重写所有使用原始op的memref.subview操作
    for (Operation *user : op->getUsers()) {
      if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
        rewriter.setInsertionPoint(subviewOp);
        // 创建新的subview，使用storeOp的base作为源
        auto newSubview = rewriter.create<memref::SubViewOp>(
            subviewOp.getLoc(),
            storeOp.getBase(),  // 新的base
            subviewOp.getMixedOffsets(),  // 保持原有offset
            subviewOp.getMixedSizes(),    // 保持原有size
            subviewOp.getMixedStrides()); // 保持原有stride
        
        // 替换原subview的所有使用
        subviewOp.getResult().replaceAllUsesWith(newSubview.getResult());
        rewriter.eraseOp(subviewOp);
      }
    }

    rewriter.eraseOp(op);

    return success();
  }
};

/// 将 iree_vector_ext.transfer_write 转换为 vector.transfer_write
struct ConvertVectorExtTransferWriteToVectorTransferWrite
    : public OpRewritePattern<IREE::VectorExt::TransferWriteOp> {
  using OpRewritePattern<IREE::VectorExt::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::VectorExt::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {
    // 获取源和结果类型
    MemRefType baseType = mlir::cast<MemRefType>(op.getBase().getType());
    MemRefType valueType = mlir::cast<MemRefType>(op.getValue().getType());

    // 创建向量类型
    VectorType vectorType =
        VectorType::get(valueType.getShape(), valueType.getElementType());

    // 获取原始的indices和mask_dims
    SmallVector<OpFoldResult> mixedIndices = op.getMixedIndices();

    // 计算mask：min(mask[dim], indices[dim] + resultShape[dim])
    Location loc = op.getLoc();

    // 创建恒等AffineMap
    auto identityMap = AffineMap::getMultiDimIdentityMap(mixedIndices.size(),
                                                         rewriter.getContext());
    auto permMapAttr = AffineMapAttr::get(identityMap);

    // 创建mask使用vector.create_mask
    Value mask = rewriter.create<vector::CreateMaskOp>(
        loc, VectorType::get(vectorType.getShape(), rewriter.getI1Type()),
        op.getMixedMaskDims());

    // 提前创建inBounds属性
    SmallVector<bool> inBoundsValues(mixedIndices.size(), true);
    auto inBoundsAttr = rewriter.getBoolArrayAttr(inBoundsValues);

    // 创建所有索引为0的ValueRange用于vector.load
    SmallVector<Value> zeroIndices;
    for (int i = 0; i < vectorType.getRank(); ++i) {
      zeroIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // 使用vector.load将value从memref转换为vector
    auto vectorValue =
        rewriter
            .create<vector::LoadOp>(loc, vectorType, op.getValue(),
                                    ValueRange(zeroIndices))
            .getResult();

    SmallVector<Value> indices;
    int dimIdx = 0;
    // 遍历所有维度
    for (int i = 0; i < valueType.getRank(); ++i) {
      if (dimIdx < op.getMixedIndices().size()) {
        auto indie = op.getMixedIndices()[dimIdx];
        if (auto attr = indie.dyn_cast<Attribute>()) {
          if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
            indices.push_back(rewriter.create<arith::ConstantIndexOp>(
                op.getLoc(), intAttr.getInt()));
          }
        } else {
          indices.push_back(cast<Value>(indie));
        }
        dimIdx++;
      } else {
        indices.push_back(
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0));
      }
    }
    // 创建vector.transfer_write操作
    auto vecWriteOp = rewriter.create<vector::TransferWriteOp>(
        loc, vectorValue, op.getBase(), indices, permMapAttr, mask,
        inBoundsAttr);

    // 删除原始的iree_vector_ext.transfer_write操作
    rewriter.replaceOp(op, vecWriteOp);

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
    patterns.add<ConvertVectorExtTransferWriteToVectorTransferWrite>(context);

    // 应用模式
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

} // namespace mlir::tts