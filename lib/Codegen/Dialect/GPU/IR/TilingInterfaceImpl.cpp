#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tts::GPU {

SmallVector<utils::IteratorType> MultiMmaOp::getLoopIteratorTypes() {
  return getIteratorTypesArray();
}

SmallVector<Range> MultiMmaOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> ranges;
  SmallVector<AffineMap> indexingMaps = getIndexingMapsArray();
  
  for (const auto &it : llvm::enumerate(getIteratorTypes())) {
    auto targetExpr = getAffineDimExpr(it.index(), builder.getContext());
    auto iteratorType = llvm::cast<IteratorTypeAttr>(it.value()).getValue();
    
    if (iteratorType == utils::IteratorType::reduction) {
      std::optional<int64_t> lhsDimIndex = indexingMaps[0].getResultPosition(targetExpr);
      assert(lhsDimIndex && "invalid lhs map");
      OpFoldResult ub = tensor::getMixedSize(builder, loc, getLhs(), *lhsDimIndex);
      ranges.emplace_back(Range{zero, ub, one});
    } else {
      std::optional<int64_t> resDimIndex = indexingMaps[2].getResultPosition(targetExpr);
      assert(resDimIndex && "invalid result map");
      OpFoldResult ub = tensor::getMixedSize(builder, loc, getAcc(), *resDimIndex);
      ranges.emplace_back(Range{zero, ub, one});
    }
  }
  return ranges;
}

static void populateSliceIndices(OpBuilder &b, Location loc, Value src,
                                 ArrayRef<OpFoldResult> offsets,
                                 ArrayRef<OpFoldResult> sizes,
                                 SmallVector<OpFoldResult> &resultOffsets,
                                 SmallVector<OpFoldResult> &resultSizes,
                                 AffineMap indexingMap) {
  int64_t srcRank = cast<RankedTensorType>(src.getType()).getRank();
  OpFoldResult zero = b.getIndexAttr(0);
  
  resultOffsets.resize(srcRank, zero);
  resultSizes.resize(srcRank, zero);

  for (auto [idx, dim] : llvm::enumerate(indexingMap.getResults())) {
    int64_t dimPos = cast<AffineDimExpr>(dim).getPosition();
    resultOffsets[idx] = offsets[dimPos];
    resultSizes[idx] = sizes[dimPos];
  }

  for (int64_t i = indexingMap.getNumResults(); i < srcRank; ++i)
    resultSizes[i] = tensor::getMixedSize(b, loc, src, i);
}

static tensor::ExtractSliceOp extractSlice(OpBuilder &b, Location loc,
                                           Value src,
                                           ArrayRef<OpFoldResult> offsets,
                                           ArrayRef<OpFoldResult> sizes,
                                           AffineMap indexingMap) {
  int64_t srcRank = cast<RankedTensorType>(src.getType()).getRank();
  SmallVector<OpFoldResult> fullOffsets(srcRank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> fullSizes(srcRank, b.getIndexAttr(0));
  
  populateSliceIndices(b, loc, src, offsets, sizes, fullOffsets, fullSizes, indexingMap);
  
  SmallVector<OpFoldResult> fullStrides(srcRank, b.getIndexAttr(1));
  return b.create<tensor::ExtractSliceOp>(loc, src, fullOffsets, fullSizes, fullStrides);
}

FailureOr<TilingResult>
MultiMmaOp::getTiledImplementation(OpBuilder &builder,
                                   ArrayRef<OpFoldResult> offsets,
                                   ArrayRef<OpFoldResult> sizes) {
  if (!hasTensorSemantics() || 
      offsets.size() != getIndexingMapsArray()[0].getNumDims() ||
      offsets.size() != sizes.size())
    return failure();

  Location loc = getLoc();
  SmallVector<Value> tiledOperands;
  SmallVector<Operation *> slices;

  // Process LHS operand
  if (Operation *lhsSlice = extractSlice(builder, loc, getLhs(), offsets, sizes, 
                                        getIndexingMapsArray()[0])) {
    tiledOperands.push_back(lhsSlice->getResult(0));
    slices.push_back(lhsSlice);
  } else return emitOpError("failed to get lhs slice");

  // Process RHS operand  
  if (Operation *rhsSlice = extractSlice(builder, loc, getRhs(), offsets, sizes,
                                        getIndexingMapsArray()[1])) {
    tiledOperands.push_back(rhsSlice->getResult(0));
    slices.push_back(rhsSlice);
  } else return emitOpError("failed to get rhs slice");

  // Process Accumulator
  if (Operation *accSlice = extractSlice(builder, loc, getAcc(), offsets, sizes,
                                        getIndexingMapsArray()[2])) {
    tiledOperands.push_back(accSlice->getResult(0));
    slices.push_back(accSlice);
  } else return emitOpError("failed to get accumulator slice");

  SmallVector<Type> resultTypes{tiledOperands.back().getType()};
  Operation *tiledMmaOp = mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{{tiledMmaOp}, {tiledMmaOp->getResults()}, slices};
}

LogicalResult MultiMmaOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0 || !hasTensorSemantics() ||
      getIndexingMapsArray()[2].getNumDims() != offsets.size() ||
      offsets.size() != sizes.size())
    return failure();

  populateSliceIndices(builder, getLoc(), getAcc(), offsets, sizes,
                      resultOffsets, resultSizes, getIndexingMapsArray()[2]);
  return success();
}

} // namespace mlir::tts::GPU
