#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::tts::IREE::VectorExt {

SmallVector<utils::IteratorType> TransferReadOp::getLoopIteratorTypes() {
  auto resultType = cast<MemRefType>(getResult().getType());
  return SmallVector<utils::IteratorType>(
      resultType.getRank(), utils::IteratorType::parallel);
}

SmallVector<Range> TransferReadOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  
  auto resultType = cast<MemRefType>(getResult().getType());
  SmallVector<Range> ranges;
  
  for (int64_t i = 0; i < resultType.getRank(); ++i) {
    Value dim;
    if (resultType.isDynamicDim(i)) {
      dim = builder.create<memref::DimOp>(loc, getResult(), i);
    } else {
      dim = builder.create<arith::ConstantIndexOp>(loc, resultType.getDimSize(i));
    }
    ranges.emplace_back(Range{zero, dim, one});
  }
  
  return ranges;
}

// 辅助函数，从 OpFoldResult 数组获取形状
static SmallVector<int64_t> getShapeFromSizes(ArrayRef<OpFoldResult> sizes) {
  SmallVector<int64_t> shape;
  for (auto size : sizes) {
    if (auto attr = size.dyn_cast<Attribute>()) {
      shape.push_back(mlir::cast<IntegerAttr>(attr).getInt());
    } else {
      shape.push_back(ShapedType::kDynamic);
    }
  }
  return shape;
}

FailureOr<TilingResult>
TransferReadOp::getTiledImplementation(OpBuilder &builder,
                                      ArrayRef<OpFoldResult> offsets,
                                      ArrayRef<OpFoldResult> sizes) {
  if (offsets.size() != sizes.size())
    return failure();

  Location loc = getLoc();
  
  // 获取原始操作的索引和掩码
  SmallVector<OpFoldResult> originalIndices = getMixedIndices();
  SmallVector<OpFoldResult> originalMaskDims = getMixedMaskDims();
  
  // 计算新的索引 (原始索引 + 偏移)
  SmallVector<OpFoldResult> newIndices;
  for (unsigned i = 0; i < offsets.size(); ++i) {
    // 如果原始索引不足,则使用0
    OpFoldResult originalIndex = i < originalIndices.size() ? 
                                originalIndices[i] : builder.getIndexAttr(0);
    
    // 计算新索引 = 原始索引 + 偏移
    Value offsetValue;
    if (auto offsetAttr = offsets[i].dyn_cast<Attribute>()) {
      offsetValue = builder.create<arith::ConstantIndexOp>(
          loc, mlir::cast<IntegerAttr>(offsetAttr).getInt());
    } else {
      offsetValue = offsets[i].dyn_cast<Value>();
    }
    
    Value originalIndexValue;
    if (auto indexAttr = originalIndex.dyn_cast<Attribute>()) {
      originalIndexValue = builder.create<arith::ConstantIndexOp>(
          loc, mlir::cast<IntegerAttr>(indexAttr).getInt());
    } else {
      originalIndexValue = originalIndex.dyn_cast<Value>();
    }
    
    Value newIndex = builder.create<arith::AddIOp>(loc, originalIndexValue, offsetValue);
    newIndices.push_back(newIndex);
  }
  
  // 创建新的结果类型 (使用tile大小)
  auto resultType = cast<MemRefType>(getResult().getType());
  auto shape = getShapeFromSizes(sizes);
  
  // 创建新的stride布局
  int64_t offset = 0;
  SmallVector<int64_t> strides;
  int64_t stride = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    strides.insert(strides.begin(), stride);
    if (shape[i] != ShapedType::kDynamic)
      stride *= shape[i];
  }
  
  auto stridedLayout = StridedLayoutAttr::get(
      builder.getContext(), offset, strides);
  
  auto newResultType = MemRefType::get(
      shape, 
      resultType.getElementType(),
      stridedLayout, 
      resultType.getMemorySpace());
  
  // 创建新的transfer_read操作
  auto newOp = builder.create<TransferReadOp>(
      loc,
      newResultType,
      getBase(),
      newIndices,
      originalMaskDims,
      getOther());
  
  return TilingResult{{newOp}, {newOp}};
}

LogicalResult 
TransferReadOp::getResultTilePosition(OpBuilder &builder,
                                     unsigned resultNumber,
                                     ArrayRef<OpFoldResult> offsets,
                                     ArrayRef<OpFoldResult> sizes,
                                     SmallVector<OpFoldResult> &resultOffsets,
                                     SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}


} // namespace mlir::tts::IREE::VectorExt