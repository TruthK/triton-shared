#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
using namespace mlir;
using namespace mlir::tts;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTTSTRANSFEROP
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// 转换 TTS_TransferReadOp 到 iree_vector_ext.transfer_read
class TransferReadOpConversion
    : public OpConversionPattern<tts::TransferReadOp> {
public:
  TransferReadOpConversion(MLIRContext *context, bool isTensorToVector = false)
      : OpConversionPattern<tts::TransferReadOp>(context),
        isTensorToVector(isTensorToVector) {}

  LogicalResult
  matchAndRewrite(tts::TransferReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isTensorToVector) {
      return matchAndRewriteToVectorTransferRead(op, adaptor, rewriter);
    } else {
      return matchAndRewriteToIREETransferRead(op, adaptor, rewriter);
    }
  }

private:
  // 转换为IREE TransferRead的实现
  LogicalResult
  matchAndRewriteToIREETransferRead(tts::TransferReadOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    // 获取操作数
    Value base = adaptor.getBase();
    auto maskDims = op.getMixedMaskDims();
    Value other = adaptor.getOther();

    // 创建默认的indices - 全0
    auto loc = op.getLoc();
    auto baseType = cast<MemRefType>(base.getType());
    int64_t rank = baseType.getRank();

    // 创建OpFoldResult数组,全部使用静态0
    SmallVector<OpFoldResult> indices;
    indices.resize(rank, rewriter.getI64IntegerAttr(0));

    // 创建带有 stride 的结果类型
    auto shape = baseType.getShape();
    int64_t offset = 0;
    SmallVector<int64_t> strides;
    int64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides.insert(strides.begin(), stride);
      if (shape[i] != ShapedType::kDynamic)
        stride *= shape[i];
    }
    auto stridedLayout =
        StridedLayoutAttr::get(rewriter.getContext(), offset, strides);

    // 创建新的 memref 类型，保持原有的地址空间
    auto resultType = MemRefType::get(shape, baseType.getElementType(),
                                      AffineMap(), baseType.getMemorySpace());

    // 使用builder创建新的操作
    auto newOp = rewriter.create<mlir::tts::IREE::VectorExt::TransferReadOp>(
        loc,
        resultType, // 带有 stride 的结果类型
        base,       // 基址
        indices,    // 索引
        maskDims,   // mask维度
        other       // 其他值
    );

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, op.getResult().getType(), newOp.getResult(), true /* restrict */,
        false /* writable */);
    rewriter.replaceOp(op, tensor);
    return success();
  }

  // 转换为vector.transfer_read的实现
  LogicalResult matchAndRewriteToVectorTransferRead(
      tts::TransferReadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const {
    // 获取操作数
    Value base = adaptor.getBase();
    auto maskDims = op.getMixedMaskDims();
    Value other = adaptor.getOther();
    auto loc = op.getLoc();

    // 获取基本信息
    auto baseType = cast<MemRefType>(base.getType());
    int64_t rank = baseType.getRank();
    auto resultType = op.getResult().getType();

    // 创建索引，默认为0
    SmallVector<Value> indices;
    for (int i = 0; i < rank; ++i) {
      indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // 获取memref的形状
    auto memrefShape = baseType.getShape();
    SmallVector<int64_t> vectorShape;
    for (auto dim : memrefShape) {
      vectorShape.push_back(dim);
    }

    // 创建vector类型作为transfer_read的返回类型
    auto vecType = VectorType::get(vectorShape, baseType.getElementType());

    // 1. 如果有掩码维度，创建vector mask
    Value mask;
    if (!maskDims.empty()) {
      // 创建所有维度的常量1作为mask的基础形状
      SmallVector<int64_t> maskShape;
      for (auto dim : memrefShape) {
        maskShape.push_back(dim);
      }

      // 创建mask的初始值 - 全部为true
      auto maskType = VectorType::get(maskShape, rewriter.getI1Type());
      Value trueMask = rewriter.create<vector::BroadcastOp>(
          loc, maskType, rewriter.create<arith::ConstantIntOp>(loc, 1, 1));

      // 对于每个维度，如果有掩码，则应用掩码
      for (int64_t i = 0; i < rank; i++) {
        if (i < static_cast<int64_t>(maskDims.size()) &&
            !maskDims[i].isNull()) {
          // 获取掩码值
          Value maskDim;
          if (auto attr = dyn_cast<Attribute>(maskDims[i])) {
            if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
              maskDim = rewriter.create<arith::ConstantOp>(loc, intAttr);
            } else {
              return failure();
            }
          } else {
            maskDim = cast<Value>(maskDims[i]);
          }

          // 为该维度创建mask
          mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, maskDim);

          // 如果有多个维度的mask，使用and操作合并
          if (trueMask) {
            mask = rewriter.create<arith::AndIOp>(loc, trueMask, mask);
          }
        }
      }

      // 如果没有创建任何维度的mask，使用全1的mask
      if (!mask) {
        mask = trueMask;
      }
    }

    // 2. 创建vector.transfer_read操作
    Value inBoundsMask;
    AffineMap map =
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    Value result;

    // 创建AffineMapAttr
    auto mapAttr = AffineMapAttr::get(map);

    // 创建in_bounds属性
    SmallVector<bool> inBounds(rank, false);
    auto inBoundsAttr = rewriter.getBoolArrayAttr(inBounds);

    if (mask) {
      // 使用mask
      result = rewriter.create<vector::TransferReadOp>(
          loc, vecType, base, indices, mapAttr, other, mask, inBoundsAttr);
    } else {
      // 不使用mask
      result = rewriter.create<vector::TransferReadOp>(
          loc, vecType, base, indices, mapAttr, inBoundsAttr);
    }

    // 3. 将vector结果转换为预期的返回类型（tensor）
    auto unrealizedCast =
        rewriter.create<UnrealizedConversionCastOp>(loc, resultType, result);

    rewriter.replaceOp(op, unrealizedCast.getResults());
    return success();
  }

  bool isTensorToVector;
};

// 转换 TTS_TransferWriteOp
class TransferWriteOpConversion
    : public OpConversionPattern<tts::TransferWriteOp> {
public:
  TransferWriteOpConversion(MLIRContext *context, bool isTensorToVector = false)
      : OpConversionPattern<tts::TransferWriteOp>(context),
        isTensorToVector(isTensorToVector) {}

  LogicalResult
  matchAndRewrite(tts::TransferWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isTensorToVector) {
      return matchAndRewriteToVectorTransferWrite(op, adaptor, rewriter);
    } else {
      return matchAndRewriteToMemrefCopy(op, adaptor, rewriter);
    }
  }

private:
  // 转换为memref.copy实现
  LogicalResult
  matchAndRewriteToMemrefCopy(tts::TransferWriteOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    // 获取操作数
    Value dest = adaptor.getBase();
    Value value = adaptor.getValue();
    auto maskDims = op.getMixedMaskDims();

    // 创建默认的indices - 全0
    auto loc = op.getLoc();
    auto baseType = cast<MemRefType>(dest.getType());
    int64_t rank = baseType.getRank();

    // 创建OpFoldResult数组，用于offsets，全部使用静态0
    SmallVector<OpFoldResult> offsets;
    offsets.resize(rank, rewriter.getI64IntegerAttr(0));

    // 获取结果的形状
    auto resultShape = baseType.getShape();

    // 创建sizes数组 - 计算min(mask[dim], resultShape[dim])
    SmallVector<OpFoldResult> sizes;
    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < rank; i++) {
      // 检查maskDims是否有该维度的掩码
      if (i < static_cast<int64_t>(maskDims.size()) && !maskDims[i].isNull()) {
        // 获取掩码值
        Value maskDim;
        if (auto attr = dyn_cast<Attribute>(maskDims[i])) {
          if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
            maskDim = rewriter.create<arith::ConstantOp>(loc, intAttr);
          } else {
            return failure();
          }
        } else {
          maskDim = cast<Value>(maskDims[i]);
        }

        // 创建常量表示resultShape[i]
        Value shapeDim;
        if (resultShape[i] == ShapedType::kDynamic) {
          // 如果是动态维度，获取动态尺寸
          auto dimOp = rewriter.create<memref::DimOp>(loc, dest, i);
          shapeDim = dimOp.getResult();
        } else {
          // 如果是静态维度，创建常量
          shapeDim =
              rewriter.create<arith::ConstantIndexOp>(loc, resultShape[i]);
        }

        // 计算min(mask[dim], resultShape[dim])
        Value minSize = rewriter.create<arith::MinSIOp>(loc, maskDim, shapeDim);
        dynamicSizes.push_back(minSize);
        sizes.push_back(minSize);
      } else {
        // 如果没有掩码，使用整个维度
        if (resultShape[i] == ShapedType::kDynamic) {
          // 如果是动态维度，获取动态尺寸
          auto dimOp = rewriter.create<memref::DimOp>(loc, dest, i);
          dynamicSizes.push_back(dimOp.getResult());
          sizes.push_back(dimOp.getResult());
        } else {
          // 如果是静态维度，使用常量
          sizes.push_back(rewriter.getIndexAttr(resultShape[i]));
        }
      }
    }

    // 创建strides数组，全部为1（表示连续存储）
    SmallVector<OpFoldResult> strides;
    strides.resize(rank, rewriter.getIndexAttr(1));

    // 1. 创建tensor.extract_slice操作
    // 获取value的类型并基于它创建extract_slice
    auto valueType = cast<TensorType>(value.getType());
    SmallVector<int64_t> sliceSizes;
    for (auto s : sizes) {
      if (auto attr = dyn_cast<Attribute>(s)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          sliceSizes.push_back(intAttr.getInt());
        } else {
          sliceSizes.push_back(ShapedType::kDynamic);
        }
      } else {
        sliceSizes.push_back(ShapedType::kDynamic);
      }
    }

    auto rankedSourceType = cast<RankedTensorType>(valueType);
    auto sliceResultType = tensor::ExtractSliceOp::inferResultType(
        rankedSourceType, SmallVector<int64_t>(rank, 0), // offsets
        sliceSizes, SmallVector<int64_t>(rank, 1)        // strides
    );

    auto extractSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, sliceResultType, value, offsets, sizes, strides);

    // 2. 使用bufferization::ToMemrefOp将切片转换为memref
    auto memrefType = MemRefType::get(
        sliceResultType.getShape(), sliceResultType.getElementType(),
        AffineMap(),                // 使用默认映射
        baseType.getMemorySpace()); // 使用和目标相同的内存空间

    auto toMemref = rewriter.create<bufferization::ToMemrefOp>(
        loc, memrefType, extractSlice.getResult(), false);

    // 3. 创建memref.subview来获取目标的相应部分
    auto destType = cast<MemRefType>(dest.getType());

    // 创建 SubViewOp
    auto resultType = memref::SubViewOp::inferResultType(destType, // 源类型
                                                         offsets,  // 偏移
                                                         sizes,    // 大小
                                                         strides   // 步长
    );

    auto subview = rewriter.create<memref::SubViewOp>(
        loc, cast<MemRefType>(resultType), dest, offsets, sizes, strides);

    // 4. 使用memref.copy将数据从源memref复制到目标memref
    rewriter.create<memref::CopyOp>(loc, toMemref, subview);

    // 5. 替换原始操作
    rewriter.eraseOp(op);
    return success();
  }

  // 转换为vector.transfer_write实现
  LogicalResult matchAndRewriteToVectorTransferWrite(
      tts::TransferWriteOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const {
    // 获取操作数
    Value dest = adaptor.getBase();
    Value value = adaptor.getValue();
    auto loc = op.getLoc();

    // 获取基本信息
    auto destType = cast<MemRefType>(dest.getType());
    int64_t rank = destType.getRank();

    // 创建索引，默认为0
    SmallVector<Value> indices;
    for (int i = 0; i < rank; ++i) {
      indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // 创建vector.transfer_write的输入值
    // 首先需要将tensor值转换为vector
    auto tensorType = cast<TensorType>(value.getType());
    auto vectorType =
        VectorType::get(tensorType.getShape(), tensorType.getElementType());

    // 将tensor转换为vector
    Value vectorValue =
        rewriter.create<UnrealizedConversionCastOp>(loc, vectorType, value)
            .getResult(0);

    // 创建vector.transfer_write操作
    // 注意：TransferWriteOp不需要掩码
    AffineMap map =
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());

    // 创建AffineMapAttr和in_bounds属性
    auto mapAttr = AffineMapAttr::get(map);
    SmallVector<bool> inBounds(rank, false);
    auto inBoundsAttr = rewriter.getBoolArrayAttr(inBounds);

    rewriter.create<vector::TransferWriteOp>(loc, vectorValue, dest, indices,
                                             mapAttr,
                                             /*mask=*/Value(), inBoundsAttr);

    // 删除原始操作
    rewriter.eraseOp(op);
    return success();
  }

  bool isTensorToVector;
};

// 修改ConvertTTSTransferOp类
class ConvertTTSTransferOp
    : public triton::impl::ConvertTTSTransferOpBase<ConvertTTSTransferOp> {
public:
  ConvertTTSTransferOp() = default;

  explicit ConvertTTSTransferOp(bool isTensorToVector)
      : isTensorToVector(isTensorToVector) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, memref::MemRefDialect,
                math::MathDialect, linalg::LinalgDialect, scf::SCFDialect,
                ttx::TritonTilingExtDialect, tts::TritonStructuredDialect,
                bufferization::BufferizationDialect, tensor::TensorDialect,
                vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    target.addLegalDialect<
        vector::VectorDialect, arith::ArithDialect, memref::MemRefDialect,
        scf::SCFDialect, math::MathDialect, linalg::LinalgDialect,
        gpu::GPUDialect, func::FuncDialect,
        IREE::VectorExt::IREEVectorExtDialect,
        bufferization::BufferizationDialect, tensor::TensorDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalOp<tts::TransferReadOp, tts::TransferWriteOp>();
    // 添加转换模式
    RewritePatternSet patterns(context);
    patterns.add<TransferReadOpConversion, TransferWriteOpConversion>(
        context, isTensorToVector);

    // 应用转换
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }

private:
  bool isTensorToVector = false;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
triton::createConvertTTSTransferOpPass() {
  return std::make_unique<ConvertTTSTransferOp>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
triton::createConvertTTSTransferOpPass(bool isTensorToVector) {
  return std::make_unique<ConvertTTSTransferOp>(isTensorToVector);
}