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

/// 添加通用的 mask 计算辅助函数
static Value createMaskFromMixedDimsAndIndices(Location loc,
                                               ArrayRef<OpFoldResult> mixedMaskDims,
                                               ValueRange indices,
                                               VectorType resultVecType,
                                               PatternRewriter &rewriter) {
  SmallVector<Value> maskValues;
  for (unsigned i = 0; i < resultVecType.getRank(); ++i) {
    // 获取mask维度值并处理越界
    Value maskDimVal;
    if (i < mixedMaskDims.size()) {
      // mixedMaskDims[i] 可能是 Attribute 或 Value
      if (auto attr = mixedMaskDims[i].dyn_cast<Attribute>()) {
        auto intAttr = cast<IntegerAttr>(attr);
        maskDimVal = rewriter.create<arith::ConstantIndexOp>(loc, intAttr.getInt());
      } else {
        // 当混合掩码维度是 Value 时，获取对应 Value
        maskDimVal = cast<Value>(mixedMaskDims[i]);
      }
    } else {
      // 如果没有提供 maskDim，默认使用 full 长度
      maskDimVal = rewriter.create<arith::ConstantIndexOp>(loc, resultVecType.getDimSize(i));
    }
    // 获取索引值
    Value idxVal = (i < indices.size()) ? indices[i]
        : rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // 获取dim大小常量
    Value dimSizeConst = rewriter.create<arith::ConstantIndexOp>(
        loc, resultVecType.getDimSize(i));
    // 计算mask值
    Value diff = rewriter.create<arith::SubIOp>(loc, maskDimVal, idxVal);
    Value minVal = rewriter.create<arith::MinSIOp>(loc, diff, dimSizeConst);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value cond = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, maskDimVal, idxVal);
    maskValues.push_back(
        rewriter.create<arith::SelectOp>(loc, cond, minVal, zero));
  }
  return rewriter.create<vector::CreateMaskOp>(
      loc, VectorType::get(resultVecType.getShape(), rewriter.getI1Type()),
      maskValues);
}

/// 优化bufferization链中的memref.copy操作
/// 将 bufferization.to_tensor -> bufferization.to_memref -> memref.copy 
/// 简化为直接 memref.copy，跳过中间转换
struct OptimizeBufferizationChainCopy
    : public OpRewritePattern<memref::CopyOp> {
public:
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    // 获取源操作数（要检查的是源操作数）
    Value source = copyOp.getSource();
    Value target = copyOp.getTarget();
    
    // 检查源是否来自 bufferization.to_memref
    auto toMemRefOp = source.getDefiningOp<bufferization::ToMemrefOp>();
    if (!toMemRefOp)
      return failure();
      
    // 检查 to_memref 的输入是否来自 bufferization.to_tensor
    Value tensorSource = toMemRefOp.getTensor();
    auto toTensorOp = tensorSource.getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensorOp)
      return failure();
      
    // 获取原始的memref源
    Value originalMemRef = toTensorOp.getMemref();
    
    // 创建新的memref.copy操作，跳过中间转换
    rewriter.replaceOpWithNewOp<memref::CopyOp>(copyOp, originalMemRef, target);
    
    return success();
  }
};

/// 将 iree_vector_ext.transfer_read 转换为 vector.transfer_read
struct ConvertVectorExtTransferReadToVectorTransferRead
    : public OpRewritePattern<IREE::VectorExt::TransferReadOp> {
public:
  using OpRewritePattern<IREE::VectorExt::TransferReadOp>::OpRewritePattern;

  LogicalResult rewriteStandaloneTransferRead(IREE::VectorExt::TransferReadOp op,
                                             PatternRewriter &rewriter) const {
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    MemRefType resultType = cast<MemRefType>(op.getResult().getType());
    VectorType vectorType =
        VectorType::get(resultType.getShape(), resultType.getElementType());
    SmallVector<OpFoldResult> mixedIndices = op.getMixedIndices();

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
    SmallVector<bool> inBoundsValues(resultType.getRank(), true);
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

    // 有条件地生成 mask，并调用相应的 builder
    vector::TransferReadOp vecReadOp;
    if (op.hasMask()) {
      Value mask = createMaskFromMixedDimsAndIndices(
          loc, op.getMixedMaskDims(), vecReadIndices,
          vectorType, rewriter);
      vecReadOp = rewriter.create<vector::TransferReadOp>(
          loc, TypeRange{vectorType}, op.getBase(),
          ValueRange(vecReadIndices), permMapAttr,
          paddingValue, mask, inBoundsAttr);
    } else {
      vecReadOp = rewriter.create<vector::TransferReadOp>(
          loc, vectorType, op.getBase(),
          ValueRange(vecReadIndices), permMapAttr,
          inBoundsAttr);
    }

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

  LogicalResult rewriteFusedTransferRead(
      IREE::VectorExt::TransferReadOp extReadOp,
      vector::TransferReadOp vecReadOp,
      PatternRewriter &rewriter) const {
    Location loc = vecReadOp.getLoc();
    rewriter.setInsertionPoint(vecReadOp);

    VectorType resultVectorType = vecReadOp.getVectorType();

    // 1. 合并 Indices
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

    // 3. 获取vector.transfer_read的其他参数
    Value base = extReadOp.getBase();
    Value padding = vecReadOp.getPadding();
    AffineMapAttr permMapAttr2 = vecReadOp.getPermutationMapAttr();
    ArrayAttr inBoundsAttr2 = vecReadOp.getInBoundsAttr();
    // 有条件地生成融合 mask 并调用合适的 builder
    vector::TransferReadOp fusedVecReadOp;
    if (extReadOp.hasMask()) {
      Value mask = createMaskFromMixedDimsAndIndices(
          loc, extReadOp.getMixedMaskDims(), combinedIndices,
          resultVectorType, rewriter);
      fusedVecReadOp = rewriter.create<vector::TransferReadOp>(
          loc, TypeRange{resultVectorType}, base,
          ValueRange(combinedIndices), permMapAttr2,
          padding, mask, inBoundsAttr2);
    } else {
      // 无mask时，只使用permutation_map和in_bounds builder
      fusedVecReadOp = rewriter.create<vector::TransferReadOp>(
          loc, resultVectorType, base,
          ValueRange(combinedIndices), permMapAttr2,
          inBoundsAttr2);
    }

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
public:
  using OpRewritePattern<IREE::VectorExt::TransferWriteOp>::OpRewritePattern;

  LogicalResult rewriteStandardTransferWrite(
      IREE::VectorExt::TransferWriteOp op,
      PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    MemRefType baseType = cast<MemRefType>(op.getBase().getType());
    MemRefType valueType = cast<MemRefType>(op.getValue().getType());
    VectorType vectorType =
        VectorType::get(valueType.getShape(), valueType.getElementType());

    // 获取原始的indices和mask_dims
    SmallVector<OpFoldResult> mixedIndices = op.getMixedIndices();
    SmallVector<OpFoldResult> mixedMaskDims = op.getMixedMaskDims();
    // 创建perm map和inBounds
    auto identityMap = AffineMap::getMultiDimIdentityMap(mixedIndices.size(), rewriter.getContext());
    auto permMapAttr = AffineMapAttr::get(identityMap);
    SmallVector<bool> inBoundsValues(valueType.getRank(), true);
    auto inBoundsAttr = rewriter.getBoolArrayAttr(inBoundsValues);

    // 创建所有索引为0的ValueRange用于vector.load
    SmallVector<Value> zeroIndices;
    for (int i = 0; i < vectorType.getRank(); ++i)
      zeroIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    // 使用vector.transfer_read将value从memref转换为vector，添加permMapAttr
    auto vectorValue = rewriter.create<vector::TransferReadOp>(
        loc, vectorType, op.getValue(), ValueRange(zeroIndices), permMapAttr, inBoundsAttr)
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
    
    // 有条件地生成 mask 并调用合适的 builder
    vector::TransferWriteOp vecWriteOp;
    if (op.hasMask()) {
      Value mask = createMaskFromMixedDimsAndIndices(
          loc, mixedMaskDims, indices,
          vectorType, rewriter);
      vecWriteOp = rewriter.create<vector::TransferWriteOp>(
          loc, vectorValue, op.getBase(),
          ValueRange(indices), permMapAttr,
          mask, inBoundsAttr);
    } else {
      // 确保inBoundsAttr包含足够的元素
      vecWriteOp = rewriter.create<vector::TransferWriteOp>(
          loc, vectorValue, op.getBase(),
          ValueRange(indices), permMapAttr,
          inBoundsAttr);  // 使用正确大小的inBoundsAttr
    }

    // 替换原始操作
    rewriter.replaceOp(op, vecWriteOp);
    return success();
  }

  LogicalResult rewriteBufferizationChain(
      IREE::VectorExt::TransferWriteOp op, Value srcMemRef,
      PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    MemRefType valueType = cast<MemRefType>(op.getValue().getType());
    VectorType vectorType =
        VectorType::get(valueType.getShape(), valueType.getElementType());

    // 获取原始的indices和mask_dims
    SmallVector<OpFoldResult> mixedIndices = op.getMixedIndices();
    SmallVector<OpFoldResult> mixedMaskDims = op.getMixedMaskDims();
    // 创建perm map和inBounds
    auto identityMap = AffineMap::getMultiDimIdentityMap(mixedIndices.size(), rewriter.getContext());
    auto permMapAttr = AffineMapAttr::get(identityMap);
    SmallVector<bool> inBoundsValues(valueType.getRank(), true);
    auto inBoundsAttr = rewriter.getBoolArrayAttr(inBoundsValues);
    // 创建所有索引为0的ValueRange用于vector.transfer_read
    SmallVector<Value> zeroIndices;
    for (int i = 0; i < vectorType.getRank(); ++i)
      zeroIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    // Zero常量作为padding
    Value elementZero = rewriter.create<arith::ConstantOp>(
        loc, vectorType.getElementType(), rewriter.getZeroAttr(vectorType.getElementType()));
    // 有条件地为 bufferization 链的读取生成 mask
    vector::TransferReadOp vectorValue;
    if (op.hasMask()) {
      Value mask = createMaskFromMixedDimsAndIndices(
          loc, mixedMaskDims, zeroIndices,
          vectorType, rewriter);
      vectorValue = rewriter.create<vector::TransferReadOp>(
          loc, TypeRange{vectorType}, srcMemRef,
          ValueRange(zeroIndices), permMapAttr,
          elementZero, /*mask=*/nullptr, inBoundsAttr);
    } else {
      // 确保inBoundsAttr与permutation_map秩匹配
      SmallVector<bool> inBoundsValues(valueType.getRank(), true);
      auto inBoundsAttr = rewriter.getBoolArrayAttr(inBoundsValues);
      
      // 使用正确的inBoundsAttr构建TransferReadOp
      vectorValue = rewriter.create<vector::TransferReadOp>(
          loc, vectorType, srcMemRef,
          ValueRange(zeroIndices), permMapAttr,
          inBoundsAttr);
    }

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

    // 使用通用函数计算mask
    Value maskVecWriteOp = createMaskFromMixedDimsAndIndices(
        loc, mixedMaskDims, targetIndices,
        vectorType, rewriter);
    // 有条件地为 bufferization 链的写入生成 mask
    vector::TransferWriteOp vecWriteOp;
    if (op.hasMask()) {
      vecWriteOp = rewriter.create<vector::TransferWriteOp>(
          loc, vectorValue.getResult(), op.getBase(),
          ValueRange(targetIndices), permMapAttr,
          maskVecWriteOp, inBoundsAttr);
    } else {
      // 确保inBoundsAttr包含足够的元素
      vecWriteOp = rewriter.create<vector::TransferWriteOp>(
          loc, vectorValue.getResult(), op.getBase(),
          ValueRange(targetIndices), permMapAttr,
          inBoundsAttr);  // 使用正确大小的inBoundsAttr
    }

    // 替换原始操作
    rewriter.replaceOp(op, vecWriteOp);
    return success();
  }

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
    // 添加新的优化模式
    patterns.add<OptimizeBufferizationChainCopy>(context);

    // 应用模式
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

} // namespace mlir::tts