#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "TransferWriteTilingInterface"

using namespace mlir;
using namespace mlir::tts;

// 前向声明TransferWriteOp
namespace mlir {
namespace tts {
class TransferWriteOp;
void registerTilingInterfaceExternalModels(DialectRegistry &registry);
} // namespace tts
} // namespace mlir

namespace {
namespace tts_impl {

// 辅助函数：从OpFoldResult中获取Value，如果是Attribute则创建常量操作
static Value getValueOrCreateConstantIndexOp(OpBuilder &b, Location loc,
                                             OpFoldResult valueOrAttr) {
  if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return b.create<arith::ConstantIndexOp>(loc, intAttr.getInt());
  }
  return valueOrAttr.dyn_cast<Value>();
}

// 将OpFoldResult数组转换为Value数组
static SmallVector<Value> convertToValues(OpBuilder &b, Location loc,
                                          ArrayRef<OpFoldResult> valueOrAttrs) {
  SmallVector<Value> values;
  values.reserve(valueOrAttrs.size());
  for (auto valueOrAttr : valueOrAttrs) {
    values.push_back(
        tts_impl::getValueOrCreateConstantIndexOp(b, loc, valueOrAttr));
  }
  return values;
}

} // namespace tts_impl

/// 实现TransferWriteOp的Tiling接口
struct TransferWriteTilingInterface
    : public TilingInterface::ExternalModel<TransferWriteTilingInterface,
                                            TransferWriteOp> {

  // 获取循环迭代器类型
  SmallVector<mlir::utils::IteratorType>
  getLoopIteratorTypes(Operation *op) const {
    // transfer_write 操作通常是并行的
    auto transferWriteOp = cast<TransferWriteOp>(op);
    auto resultType = transferWriteOp.getValue().getType();
    auto shapedType = cast<ShapedType>(resultType);

    // 为每个维度生成一个并行迭代器类型
    return SmallVector<mlir::utils::IteratorType>(
        shapedType.getRank(), mlir::utils::IteratorType::parallel);
  }

  // 获取迭代域
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto transferWriteOp = cast<TransferWriteOp>(op);
    auto loc = transferWriteOp.getLoc();
    auto resultType = transferWriteOp.getValue().getType();
    auto shapedType = cast<ShapedType>(resultType);

    SmallVector<Range> ranges;
    ranges.reserve(shapedType.getRank());

    for (int64_t i = 0; i < shapedType.getRank(); ++i) {
      Value dim = b.create<tensor::DimOp>(loc, transferWriteOp.getValue(), i);
      Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
      Value one = b.create<arith::ConstantIndexOp>(loc, 1);
      ranges.push_back(Range{zero, dim, one});
    }

    return ranges;
  }

  // 实现tiled操作
  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto transferWriteOp = cast<TransferWriteOp>(op);
    auto loc = transferWriteOp.getLoc();

    // 创建新的indices，基于原始indices和tile的offsets
    SmallVector<OpFoldResult> newIndices;
    auto dynamicIndices = transferWriteOp.getIndices();
    auto staticIndices = transferWriteOp.getStaticIndices();

    // 合并当前indices和offsets
    for (auto i = 0; i < offsets.size(); ++i) {
      // 获取原始索引
      OpFoldResult origIndex;
      if (i < dynamicIndices.size()) {
        origIndex = dynamicIndices[i];
      } else {
        int64_t staticVal = staticIndices[i];
        origIndex = b.getIndexAttr(staticVal);
      }

      // 合并原始索引和offset
      Value offsetValue =
          tts_impl::getValueOrCreateConstantIndexOp(b, loc, offsets[i]);
      Value origIndexValue =
          tts_impl::getValueOrCreateConstantIndexOp(b, loc, origIndex);
      Value newIndex =
          b.create<arith::AddIOp>(loc, origIndexValue, offsetValue);
      newIndices.push_back(newIndex);
    }

    // 从原始value创建切片
    Value originalValue = transferWriteOp.getValue();
    ShapedType originalType = cast<ShapedType>(originalValue.getType());

    // 创建新的结果类型（基于tile大小）
    SmallVector<int64_t> newShape;
    for (auto size : sizes) {
      if (auto attr = size.dyn_cast<Attribute>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          newShape.push_back(intAttr.getInt());
        }
      } else {
        // 如果大小是动态的，使用动态维度
        newShape.push_back(ShapedType::kDynamic);
      }
    }

    auto newType =
        RankedTensorType::get(newShape, originalType.getElementType());

    // 创建一个extract_slice操作来获取原始value的切片
    SmallVector<OpFoldResult> strides(sizes.size(), b.getIndexAttr(1));
    Value extractedValue = b.create<tensor::ExtractSliceOp>(
        loc, newType, originalValue, offsets, sizes, strides);

    // 创建新的TransferWriteOp
    // 将OpFoldResult转换为Value
    SmallVector<Value> indicesValues =
        tts_impl::convertToValues(b, loc, newIndices);

    // 获取mask维度，将OperandRange转换为Values
    SmallVector<Value> maskDimValues;
    for (auto maskDim : transferWriteOp.getMaskDims()) {
      maskDimValues.push_back(maskDim);
    }

    // 创建新的操作
    auto newOp = b.create<TransferWriteOp>(
        loc, transferWriteOp.getBase(), extractedValue, indicesValues,
        ArrayRef<int64_t>{}, maskDimValues, ArrayRef<int64_t>{});

    // 返回tiling结果
    TilingResult result;
    result.tiledOps.push_back(newOp);
    return result;
  }

  // 获取结果tile位置
  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    return success();
  }
};

} // namespace

void mlir::tts::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, TritonStructuredDialect *dialect) {
        TransferWriteOp::attachInterface<TransferWriteTilingInterface>(*ctx);
      });
}
