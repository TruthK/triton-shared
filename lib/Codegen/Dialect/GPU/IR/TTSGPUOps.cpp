#ifndef TRITON_SHARED_CODEGEN_DIALECT_TTS_GPU_OPS
#define TRITON_SHARED_CODEGEN_DIALECT_TTS_GPU_OPS

#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUOps.h"
#include <functional>
#include <numeric>

#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUAttrs.h"
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUDialect.h"
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#define GET_OP_CLASSES
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUOps.cpp.inc" // IWYU pragma: keep

namespace mlir::tts::GPU {

void BarrierRegionOp::build(OpBuilder &b, OperationState &result,
                            TypeRange resultTypes, ValueRange inputs) {
  result.addOperands(inputs);
  (void)result.addRegion();
  result.addTypes(resultTypes);
  SmallVector<Location> blockArgLocs(inputs.size(), result.location);

  Region *region = result.regions[0].get();
  OpBuilder::InsertionGuard guard(b);
  b.createBlock(region, region->end(), inputs.getTypes(), blockArgLocs);
}

LogicalResult BarrierRegionOp::verify() { return success(); }

LogicalResult BarrierRegionOp::verifyRegions() {
  auto &region = getRegion();
  Block &block = region.front();
  if (block.getNumArguments() != getNumOperands())
    return emitError("expected block argument count to match operand count");

  if (!llvm::all_of_zip(block.getArgumentTypes(), getOperandTypes(),
                        [](Type a, Type b) { return a == b; }))
    return emitError("expected block argument types to match operand types");

  auto yieldOp = llvm::cast<GPU::YieldOp>(block.getTerminator());
  if (yieldOp->getNumOperands() != getNumResults())
    return emitOpError("expected body to yield same number of values as results");

  if (!llvm::all_of_zip(yieldOp->getOperandTypes(), getResultTypes(),
                        [](Type a, Type b) { return a == b; }))
    return emitError("expected yielded value types to match result types");

  return success();
}

void MultiMmaOp::build(OpBuilder &builder, OperationState &result, Value lhs,
                       Value rhs, Value acc, ArrayRef<AffineMap> indexingMaps,
                       ArrayRef<utils::IteratorType> iteratorTypes,
                       MmaInterfaceAttr kind,
                       std::optional<SmallVector<int64_t>> lhsPermutation,
                       std::optional<SmallVector<int64_t>> rhsPermutation,
                       std::optional<SmallVector<int64_t>> accPermutation) {
  result.addOperands({lhs, rhs, acc});
  result.addTypes(acc.getType());
  result.addAttribute(getIndexingMapsAttrName(result.name),
                      builder.getAffineMapArrayAttr(indexingMaps));
  result.addAttribute(
      getIteratorTypesAttrName(result.name),
      builder.getArrayAttr(llvm::to_vector(llvm::map_range(
          iteratorTypes, [&](utils::IteratorType t) -> mlir::Attribute {
            return IteratorTypeAttr::get(builder.getContext(), t);
          }))));
  result.addAttribute(getKindAttrName(result.name), kind);
  if (lhsPermutation)
    result.addAttribute(getLhsPermutationAttrName(result.name),
                        builder.getDenseI64ArrayAttr(*lhsPermutation));
  if (rhsPermutation)
    result.addAttribute(getRhsPermutationAttrName(result.name),
                        builder.getDenseI64ArrayAttr(*rhsPermutation));
  if (accPermutation)
    result.addAttribute(getAccPermutationAttrName(result.name),
                        builder.getDenseI64ArrayAttr(*accPermutation));
}

void MultiMmaOp::build(OpBuilder &builder, OperationState &result, Value lhs,
                       Value rhs, Value acc,
                       ArrayRef<ArrayRef<AffineExpr>> indexingExprs,
                       ArrayRef<utils::IteratorType> iteratorTypes,
                       MmaInterfaceAttr kind,
                       std::optional<SmallVector<int64_t>> lhsPermutation,
                       std::optional<SmallVector<int64_t>> rhsPermutation,
                       std::optional<SmallVector<int64_t>> accPermutation) {
  build(builder, result, lhs, rhs, acc,
        AffineMap::inferFromExprList(indexingExprs, builder.getContext()),
        iteratorTypes, kind, lhsPermutation, rhsPermutation, accPermutation);
}

void MultiMmaOp::build(OpBuilder &builder, OperationState &result, Value lhs,
                       Value rhs, Value acc, ArrayAttr indexingMaps,
                       ArrayAttr iteratorTypes, MmaInterfaceAttr kind,
                       std::optional<DenseI64ArrayAttr> lhsPermutation,
                       std::optional<DenseI64ArrayAttr> rhsPermutation,
                       std::optional<DenseI64ArrayAttr> accPermutation) {
  result.addOperands({lhs, rhs, acc});
  result.addTypes(acc.getType());
  result.addAttribute(getIndexingMapsAttrName(result.name), indexingMaps);
  result.addAttribute(getIteratorTypesAttrName(result.name), iteratorTypes);
  result.addAttribute(getKindAttrName(result.name), kind);
  if (lhsPermutation)
    result.addAttribute(getLhsPermutationAttrName(result.name), *lhsPermutation);
  if (rhsPermutation)
    result.addAttribute(getRhsPermutationAttrName(result.name), *rhsPermutation);
  if (accPermutation)
    result.addAttribute(getAccPermutationAttrName(result.name), *accPermutation);
}

static int64_t multiplyAcc(ArrayRef<int64_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), 1,
                         std::multiplies<int64_t>());
}

LogicalResult MultiMmaOp::verify() {
  ShapedType lhsType = getLhsType();
  ShapedType rhsType = getRhsType();
  ShapedType accType = getAccType();
  SmallVector<AffineMap, 4> indexingMaps = getIndexingMapsArray();

  if (indexingMaps.size() != 3)
    return emitOpError("expected an indexing map for each operand");

  unsigned numIterators = getIteratorTypes().getValue().size();
  for (const auto &it : llvm::enumerate(indexingMaps)) {
    auto map = it.value();
    if (map.getNumSymbols() != 0)
      return emitOpError("expected indexing map ")
             << it.index() << " to have no symbols";
    auto shapedType = llvm::dyn_cast<ShapedType>(getOperand(it.index()).getType());
    unsigned rank = shapedType.getRank();
    if (map.getNumDims() != numIterators)
      return emitOpError("expected indexing map ")
             << it.index() << " to have " << numIterators << " inputs";
    if (map.getNumResults() >= rank)
      return emitOpError("expected indexing map ")
             << it.index() << " to have fewer than " << rank << " outputs";
    if (!map.isProjectedPermutation())
      return emitOpError("expected indexing map ")
             << it.index() << " to be a projected permutation";
  }

  if (failed(linalg::inferContractionDims(indexingMaps)))
    return emitOpError("failed to infer contraction dims");

  SmallVector<int64_t> bounds;
  getIterationBounds(bounds);
  auto verifyOperandShape = [&](ShapedType type, AffineMap map) {
    for (auto [dim, size] : llvm::zip(map.getResults(), type.getShape())) {
      int64_t dimIdx = cast<AffineDimExpr>(dim).getPosition();
      if (size != bounds[dimIdx]) return failure();
    }
    return success();
  };
  if (failed(verifyOperandShape(lhsType, indexingMaps[0])) ||
      failed(verifyOperandShape(rhsType, indexingMaps[1])) ||
      failed(verifyOperandShape(accType, indexingMaps[2])))
    return emitOpError("shape does not match iteration bounds");

  auto [lType, rType, aType] = getKind().getABCElementTypes();
  if (lType != lhsType.getElementType() ||
      rType != rhsType.getElementType() ||
      aType != accType.getElementType())
    return emitOpError("element type mismatch for intrinsic");

  int64_t lhsInner = multiplyAcc(getLhsInnerShape());
  int64_t rhsInner = multiplyAcc(getRhsInnerShape());
  int64_t accInner = multiplyAcc(getAccInnerShape());
  auto [m, n, k] = getKind().getMNKShape();

  if ((m * k != lhsInner || n * k != rhsInner || m * n != accInner) &&
      (getLhsPermutation() || getRhsPermutation() || getAccPermutation()))
    return emitOpError("permutations require subgroup semantics");

  if ((getLhsPermutation() && !isPermutationVector(*getLhsPermutation())) ||
      (getRhsPermutation() && !isPermutationVector(*getRhsPermutation())) ||
      (getAccPermutation() && !isPermutationVector(*getAccPermutation())))
    return emitOpError("invalid permutation vector");

  return success();
}

bool MultiMmaOp::hasThreadSemantics() {
  auto [m, n, k] = getKind().getMNKShape();
  return m * k != multiplyAcc(getLhsInnerShape()) ||
         n * k != multiplyAcc(getRhsInnerShape()) ||
         m * n != multiplyAcc(getAccInnerShape());
}

static int64_t getResultIndex(AffineMap map, AffineExpr targetExpr) {
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i)
    if (targetExpr == map.getResult(i)) return i;
  return -1;
}

void MultiMmaOp::getIterationBounds(SmallVectorImpl<int64_t> &iterationBounds) {
  auto lhsShape = getLhsType().getShape();
  auto resType = getResultType();
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMapsArray());
  
  for (const auto &it : llvm::enumerate(getIteratorTypes())) {
    auto targetExpr = getAffineDimExpr(it.index(), getContext());
    auto iteratorType = llvm::cast<IteratorTypeAttr>(it.value()).getValue();
    if (iteratorType == utils::IteratorType::reduction) {
      int64_t lhsDimIndex = getResultIndex(indexingMaps[0], targetExpr);
      iterationBounds.push_back(lhsShape[lhsDimIndex]);
    } else {
      int64_t resDimIndex = getResultIndex(indexingMaps[2], targetExpr);
      iterationBounds.push_back(resType.getShape()[resDimIndex]);
    }
  }
}

std::optional<SmallVector<int64_t, 4>> MultiMmaOp::getShapeForUnroll() {
  SmallVector<int64_t, 4> shape;
  getIterationBounds(shape);
  return shape;
}

void ValueBarrierOp::build(OpBuilder &builder, OperationState &result,
                           ValueRange input) {
  result.addOperands(input);
  result.addTypes(llvm::map_range(input, [](Value v) { return v.getType(); }));
}

LogicalResult ValueBarrierOp::verify() {
  if (getNumOperands() == 0)
    return emitOpError("Atleast one input required");

  bool allTensor = llvm::all_of(getInputTypes(), 
                               llvm::IsaPred<RankedTensorType>);
  bool allVector = llvm::all_of(getInputTypes(),
                               llvm::IsaPred<VectorType>);
  if (!(allTensor || allVector))
    return emitOpError("All inputs must be either tensor or vector type");
  
  return success();
}

} // namespace mlir::tts::GPU
