// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "triton-shared/Codegen/Common/Passes.h"
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Matchers.h"

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
private:
  // 计算 mask 的辅助函数
  Value calculateMask(IREE::VectorExt::TransferReadOp op,
                      VectorType resultVectorType,
                      PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    SmallVector<Value> maskValues;
    SmallVector<OpFoldResult> mixedMaskDims = op.getMixedMaskDims();

    for (unsigned i = 0; i < mixedMaskDims.size(); ++i) {
      OpFoldResult maskDim = mixedMaskDims[i];
      int64_t resultDimSize = resultVectorType.getDimSize(i);

      // 获取mask维度的值
      Value maskDimValue;
      if (auto attr = maskDim.dyn_cast<Attribute>()) {
        maskDimValue = rewriter.create<arith::ConstantIndexOp>(
            loc, mlir::cast<IntegerAttr>(attr).getInt());
      } else {
        maskDimValue = cast<Value>(maskDim);
      }

      // 获取indices维度的值
      Value indexValue;
      if (i < op.getIndices().size()) {
        indexValue = op.getIndices()[i];
      } else {
        indexValue = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      }

      // 计算 indices[dim] + resultShape[dim]
      Value resultDimSizeValue =
          rewriter.create<arith::ConstantIndexOp>(loc, resultDimSize);
      Value indicesPlusSize =
          rewriter.create<arith::AddIOp>(loc, indexValue, resultDimSizeValue);

      // 计算 RemSIOp（min(mask[dim], indices[dim] + resultShape[dim])，resultShape[dim] ）
      Value minVal =
          rewriter.create<arith::MinSIOp>(loc, maskDimValue, indicesPlusSize);
      Value remVal =
          rewriter.create<arith::RemSIOp>(loc, minVal, resultDimSizeValue);
      maskValues.push_back(remVal);
    }

    return rewriter.create<vector::CreateMaskOp>(
        loc, VectorType::get(resultVectorType.getShape(), rewriter.getI1Type()),
        maskValues);
  }

  // 将独立的 iree_vector_ext.transfer_read 转换为 vector.transfer_read + memref.alloc + vector.transfer_write
  LogicalResult rewriteStandaloneTransferRead(IREE::VectorExt::TransferReadOp op,
                                             PatternRewriter &rewriter) const {
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    MemRefType resultType = mlir::cast<MemRefType>(op.getResult().getType());
    VectorType vectorType =
        VectorType::get(resultType.getShape(), resultType.getElementType());
    SmallVector<OpFoldResult> mixedIndices = op.getMixedIndices();

    // 计算mask
    Value mask = calculateMask(op, vectorType, rewriter);

    // 获取paddingValue
    Value paddingValue = op.getOther();
    if (!paddingValue) {
      Type elementType = resultType.getElementType();
      paddingValue = rewriter.create<arith::ConstantOp>(
          loc, elementType, rewriter.getZeroAttr(elementType));
    }

    // 创建恒等AffineMap
    auto identityMap = AffineMap::getMultiDimIdentityMap(mixedIndices.size(),
                                                         rewriter.getContext());
    auto permMapAttr = AffineMapAttr::get(identityMap);

    // 创建inBounds属性
    SmallVector<bool> inBoundsValues(mixedIndices.size(), true);
    auto inBoundsAttr = rewriter.getBoolArrayAttr(inBoundsValues);

    // 准备 vector.transfer_read 的索引
    SmallVector<Value> vecReadIndices;
    int dimIdx = 0;
    for (int i = 0; i < resultType.getRank(); ++i) {
       // 注意：这里假设 op.getMixedIndices() 的大小与 resultType.getRank() 匹配
       // 或者说，iree_vector_ext.transfer_read 的索引维度数应该等于结果 memref 的秩
       // 如果不匹配，需要调整逻辑
      if (dimIdx < op.getMixedIndices().size()) {
        auto indie = op.getMixedIndices()[dimIdx];
        if (auto attr = indie.dyn_cast<Attribute>()) {
          if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
            vecReadIndices.push_back(rewriter.create<arith::ConstantIndexOp>(
                loc, intAttr.getInt()));
          } else {
             // Handle other attribute types if necessary
             return op.emitError("Unsupported attribute type for index");
          }
        } else {
          vecReadIndices.push_back(cast<Value>(indie));
        }
        dimIdx++;
      } else {
         // 如果索引数量少于秩，用 0 填充，但这通常表示有问题
         // 考虑是否应该报错或有不同的处理方式
        vecReadIndices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }
    }

    // 创建 vector.transfer_read
    auto vecReadOp = rewriter.create<vector::TransferReadOp>(
        loc, vectorType, op.getBase(), vecReadIndices, permMapAttr, paddingValue,
        mask, inBoundsAttr);

    // 创建 memref.alloc
    auto addressSpace = gpu::AddressSpaceAttr::get(
        rewriter.getContext(), gpu::GPUDialect::getPrivateAddressSpace());
    auto allocType =
        MemRefType::get(resultType.getShape(), resultType.getElementType(),
                        AffineMap(), addressSpace);
    auto allocOp = rewriter.create<memref::AllocOp>(loc, allocType);

    // 创建全零索引用于 vector.transfer_write
    SmallVector<Value> zeroIndices(resultType.getRank(),
                                   rewriter.create<arith::ConstantIndexOp>(loc, 0));

    // 创建 vector.transfer_write
    auto storeOp = rewriter.create<vector::TransferWriteOp>(
        loc, vecReadOp.getResult(), allocOp.getResult(),
        ValueRange(zeroIndices), permMapAttr, inBoundsAttr);

    // 替换原始 op 的用途
    op.getResult().replaceAllUsesWith(storeOp.getSource()); // 直接替换，不需要再检查 user 类型

    // 删除原始op
    rewriter.eraseOp(op);
    return success();
  }

  // 将 iree_vector_ext.transfer_read + vector.transfer_read 融合成一个 vector.transfer_read
  LogicalResult rewriteFusedTransferRead(IREE::VectorExt::TransferReadOp extReadOp,
                                         vector::TransferReadOp vecReadOp,
                                         PatternRewriter &rewriter) const {
    Location loc = vecReadOp.getLoc(); // Use the location of the consuming op
    rewriter.setInsertionPoint(vecReadOp);

    VectorType resultVectorType = vecReadOp.getVectorType();

    // 1. 计算 Mask (使用 extReadOp 的 mask_dims 和 indices, 以及最终的 resultVectorType)
    Value mask = calculateMask(extReadOp, resultVectorType, rewriter);

    // 2. 合并 Indices
    SmallVector<Value> combinedIndices;
    ValueRange extIndices = extReadOp.getIndices();
    ValueRange vecIndices = vecReadOp.getIndices();
    unsigned maxRank = std::max(extIndices.size(), vecIndices.size());

    for (unsigned i = 0; i < maxRank; ++i) {
        Value extIdx = (i < extIndices.size()) ? extIndices[i] : rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value vecIdx = (i < vecIndices.size()) ? vecIndices[i] : rewriter.create<arith::ConstantIndexOp>(loc, 0);

        // 如果 vecIdx 是常量 0，则结果就是 extIdx
        IntegerAttr vecIdxAttr;
        if (matchPattern(vecIdx, m_Constant(&vecIdxAttr)) && vecIdxAttr.getInt() == 0) {
             combinedIndices.push_back(extIdx);
        } else {
            // 否则，创建加法操作
             combinedIndices.push_back(rewriter.create<arith::AddIOp>(loc, extIdx, vecIdx));
        }
    }


    // 3. 获取其他参数
    Value base = extReadOp.getBase(); // Base 来自 extReadOp
    Value padding = vecReadOp.getPadding(); // Padding 来自 vecReadOp
    AffineMapAttr permutationMap = vecReadOp.getPermutationMapAttr(); // Permutation map 来自 vecReadOp
    ArrayAttr inBounds = vecReadOp.getInBoundsAttr(); // in_bounds 来自 vecReadOp

    // 4. 创建新的 vector.transfer_read
    auto fusedVecReadOp = rewriter.create<vector::TransferReadOp>(
        loc, resultVectorType, base, combinedIndices, permutationMap, padding,
        mask, inBounds);

    // 5. 替换并删除旧的操作
    rewriter.replaceOp(vecReadOp, fusedVecReadOp.getResult());
    // 如果 extReadOp 没有其他用户，则可以安全删除
    if (extReadOp->use_empty()) {
        rewriter.eraseOp(extReadOp);
    } else {
       // 如果 extReadOp 还有其他用户，则不能删除，融合可能不安全或需要更复杂的处理
       return extReadOp.emitWarning("extReadOp has other users, fusion might be incomplete or unsafe.");
    }


    return success();
  }

public:
  using OpRewritePattern<IREE::VectorExt::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::VectorExt::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    // 检查唯一用户是否为 vector.transfer_read
    if (op.getResult().hasOneUse()) {
      if (auto vecReadUser =
              dyn_cast<vector::TransferReadOp>(*op.getResult().getUsers().begin())) {
         // 检查 vecReadUser 的 source 是否确实是 op 的 result
         if (vecReadUser.getSource() == op.getResult()){
            return rewriteFusedTransferRead(op, vecReadUser, rewriter);
         }
      }
    }

    // 默认情况：调用独立的转换逻辑
    return rewriteStandaloneTransferRead(op, rewriter);
  }
};

/// 将 iree_vector_ext.transfer_write 转换为 vector.transfer_write
struct ConvertVectorExtTransferWriteToVectorTransferWrite
    : public OpRewritePattern<IREE::VectorExt::TransferWriteOp> {
private:
  // 计算 mask 的辅助函数
  Value calculateMask(IREE::VectorExt::TransferWriteOp op,
                      VectorType vectorType,
                      PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    return rewriter.create<vector::CreateMaskOp>(
        loc, VectorType::get(vectorType.getShape(), rewriter.getI1Type()),
        op.getMixedMaskDims());
  }

  // 处理标准的 transfer_write 转换
  LogicalResult rewriteStandardTransferWrite(IREE::VectorExt::TransferWriteOp op,
                                            PatternRewriter &rewriter) const {
    // 获取源和结果类型
    MemRefType baseType = mlir::cast<MemRefType>(op.getBase().getType());
    MemRefType valueType = mlir::cast<MemRefType>(op.getValue().getType());

    // 创建向量类型
    VectorType vectorType =
        VectorType::get(valueType.getShape(), valueType.getElementType());

    // 获取原始的indices和mask_dims
    SmallVector<OpFoldResult> mixedIndices = op.getMixedIndices();
    Location loc = op.getLoc();

    // 创建恒等AffineMap
    auto identityMap = AffineMap::getMultiDimIdentityMap(mixedIndices.size(),
                                                         rewriter.getContext());
    auto permMapAttr = AffineMapAttr::get(identityMap);

    // 创建mask
    Value mask = calculateMask(op, vectorType, rewriter);

    // 提前创建inBounds属性
    SmallVector<bool> inBoundsValues(mixedIndices.size(), true);
    auto inBoundsAttr = rewriter.getBoolArrayAttr(inBoundsValues);

    // 创建所有索引为0的ValueRange用于vector.load
    SmallVector<Value> zeroIndices;
    for (int i = 0; i < vectorType.getRank(); ++i) {
      zeroIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // 使用vector.transfer_read将value从memref转换为vector
    auto vectorValue =
        rewriter
            .create<vector::TransferReadOp>(loc, vectorType, op.getValue(),
                                            ValueRange(zeroIndices),
                                            permMapAttr, inBoundsAttr)
            .getResult();

    // 准备索引
    SmallVector<Value> indices;
    int dimIdx = 0;
    // 遍历所有维度
    for (int i = 0; i < valueType.getRank(); ++i) {
      if (dimIdx < op.getMixedIndices().size()) {
        auto indie = op.getMixedIndices()[dimIdx];
        if (auto attr = indie.dyn_cast<Attribute>()) {
          if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
            indices.push_back(rewriter.create<arith::ConstantIndexOp>(
                loc, intAttr.getInt()));
          }
        } else {
          indices.push_back(cast<Value>(indie));
        }
        dimIdx++;
      } else {
        indices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }
    }
    
    // 创建vector.transfer_write操作
    auto vecWriteOp = rewriter.create<vector::TransferWriteOp>(
        loc, vectorValue, op.getBase(), indices, permMapAttr, mask,
        inBoundsAttr);

    // 替换原始操作
    rewriter.replaceOp(op, vecWriteOp);
    return success();
  }

  // 处理优化的缓冲区链路模式:
  // bufferization.to_tensor -> buffer.to_memref -> iree_vector_ext.transfer_write
  LogicalResult rewriteBufferizationChain(IREE::VectorExt::TransferWriteOp op,
                                           Value srcMemRef,
                                           PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    MemRefType baseType = mlir::cast<MemRefType>(op.getBase().getType());
    MemRefType valueType = mlir::cast<MemRefType>(op.getValue().getType());
    VectorType vectorType =
        VectorType::get(valueType.getShape(), valueType.getElementType());

    // 创建恒等AffineMap
    auto identityMap = AffineMap::getMultiDimIdentityMap(op.getMixedIndices().size(),
                                                         rewriter.getContext());
    auto permMapAttr = AffineMapAttr::get(identityMap);

    // 创建mask
    Value mask = calculateMask(op, vectorType, rewriter);

    // 创建inBounds属性
    SmallVector<bool> inBoundsValues(op.getMixedIndices().size(), true);
    auto inBoundsAttr = rewriter.getBoolArrayAttr(inBoundsValues);

    // 创建所有索引为0的ValueRange用于vector.transfer_read
    SmallVector<Value> zeroIndices;
    for (int i = 0; i < vectorType.getRank(); ++i) {
      zeroIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }
    
    // 创建Zero常量作为padding
    Value elementZero = rewriter.create<arith::ConstantOp>(
        loc, vectorType.getElementType(), 
        rewriter.getZeroAttr(vectorType.getElementType()));

    // 直接从源memref读取向量
    auto vectorValue = rewriter.create<vector::TransferReadOp>(
        loc, vectorType, srcMemRef, zeroIndices, permMapAttr, elementZero,
        nullptr, inBoundsAttr);

    // 准备目标索引
    SmallVector<Value> targetIndices;
    int dimIdx = 0;
    for (int i = 0; i < valueType.getRank(); ++i) {
      if (dimIdx < op.getMixedIndices().size()) {
        auto indie = op.getMixedIndices()[dimIdx];
        if (auto attr = indie.dyn_cast<Attribute>()) {
          if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
            targetIndices.push_back(rewriter.create<arith::ConstantIndexOp>(
                loc, intAttr.getInt()));
          }
        } else {
          targetIndices.push_back(cast<Value>(indie));
        }
        dimIdx++;
      } else {
        targetIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }
    }

    // 创建vector.transfer_write操作
    auto vecWriteOp = rewriter.create<vector::TransferWriteOp>(
        loc, vectorValue.getResult(), op.getBase(), targetIndices, permMapAttr, mask,
        inBoundsAttr);

    // 替换原始操作
    rewriter.replaceOp(op, vecWriteOp);
    return success();
  }

public:
  using OpRewritePattern<IREE::VectorExt::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::VectorExt::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {
    // 检查是否符合优化模式：
    // 1. value 来自 bufferization.to_memref
    // 2. to_memref 的输入来自 bufferization.to_tensor
    // 3. 找到原始的 memref 源
    Value valueOperand = op.getValue();
    Operation *definingOp = valueOperand.getDefiningOp();
    
    // 尝试将definingOp转换为ToMemrefOp
    auto toMemRefOp = dyn_cast_or_null<bufferization::ToMemrefOp>(definingOp);
    if (toMemRefOp) {
      Value tensorSource = toMemRefOp.getMemref();
      Operation *tensorDefiningOp = tensorSource.getDefiningOp();
      
      // 尝试将tensorDefiningOp转换为ToTensorOp
      auto toTensorOp = dyn_cast_or_null<bufferization::ToTensorOp>(tensorDefiningOp);
      if (toTensorOp) {
        // 获取原始memref
        Value srcMemRef = toTensorOp.getMemref();
        return rewriteBufferizationChain(op, srcMemRef, rewriter);
      }
    }

    // 默认情况：使用标准的转换
    return rewriteStandardTransferWrite(op, rewriter);
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