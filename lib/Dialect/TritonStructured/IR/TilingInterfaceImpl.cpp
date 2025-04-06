#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::tts {

//===----------------------------------------------------------------------===//
// TransferReadOp
//===----------------------------------------------------------------------===//

SmallVector<::mlir::utils::IteratorType> TransferReadOp::getLoopIteratorTypes() {
  // TransferReadOp的迭代器类型都是并行的
  auto resultType = cast<RankedTensorType>(getResult().getType());
  return SmallVector<::mlir::utils::IteratorType>(resultType.getRank(),
                                         ::mlir::utils::IteratorType::parallel);
}

SmallVector<Range> TransferReadOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> ranges;
  
  auto resultType = cast<RankedTensorType>(getResult().getType());
  for (int64_t i = 0; i < resultType.getRank(); ++i) {
    OpFoldResult ub = tensor::getMixedSize(builder, loc, getResult(), i);
    ranges.emplace_back(Range{zero, ub, one});
  }
  return ranges;
}

FailureOr<TilingResult>
TransferReadOp::getTiledImplementation(OpBuilder &builder,
                                      ArrayRef<OpFoldResult> offsets,
                                      ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();
  auto resultType = cast<RankedTensorType>(getResult().getType());
  
  // 创建新的结果类型
  SmallVector<int64_t> tileShape;
  for (const auto &size : sizes) {
    if (auto attr = dyn_cast<IntegerAttr>(size.dyn_cast<Attribute>())) {
      tileShape.push_back(attr.getInt());
    } else {
      return failure();
    }
  }
  auto tileResultType = RankedTensorType::get(tileShape, resultType.getElementType());

  // 创建新的TransferReadOp
  SmallVector<Value> dynamicIndices;
  SmallVector<int64_t> staticIndices;
  dispatchIndexOpFoldResults(offsets, dynamicIndices, staticIndices);

  SmallVector<Value> dynamicMaskDims;
  SmallVector<int64_t> staticMaskDims;
  dispatchIndexOpFoldResults(getMixedMaskDims(), dynamicMaskDims, staticMaskDims);

  Operation *tiledOp = builder.create<TransferReadOp>(
      loc, tileResultType, getBase(), dynamicIndices,
      builder.getDenseI64ArrayAttr(staticIndices), dynamicMaskDims,
      builder.getDenseI64ArrayAttr(staticMaskDims), getOther());

  return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults()), {}};
}

LogicalResult TransferReadOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0) {
    return failure();
  }

  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

} // namespace mlir::tts