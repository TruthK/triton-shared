//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR//MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#define DEBUG_TYPE "structured-to-memref"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

static const std::string WRAP_SIDE_BY_SIDE = "wrap_side_by_side";
static const std::string WRAP_STACKED = "wrap_stacked";

static memref::SubViewOp getSubview(int rank, ArrayRef<OpFoldResult> dims,
                                    Value source, Location loc, OpBuilder &b) {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, offsets, dims, strides);

  return b.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType), source,
                                     offsets, dims, strides);
}

namespace {

struct MakeTensorPtrConverter
    : public OpConversionPattern<tts::MakeTensorPtrOp> {
private:
  using OpConversionPattern<tts::MakeTensorPtrOp>::OpConversionPattern;

  static Type getElementTypeStructuredPtr(tts::MakeTensorPtrOp op) {
    assert(!op.isBlockPtr());
    // tensor<1024x!tt.ptr<f32>>
    auto ptrType = cast<triton::PointerType>(
        cast<RankedTensorType>(op.getType()).getElementType());
    return ptrType.getPointeeType();
  }

  static Type getElementTypeBlockPtr(tts::MakeTensorPtrOp op) {
    assert(op.isBlockPtr());
    // !tt.ptr<tensor<128x64xbf16>, 1>
    auto shapedType = cast<ShapedType>(
        cast<triton::PointerType>(op.getType()).getPointeeType());
    return shapedType.getElementType();
  }

  static MemRefType getResultMemrefType(tts::MakeTensorPtrOp op, int64_t offset,
                                        ArrayRef<int64_t> staticStrides,
                                        ArrayRef<int64_t> resultShape) {
    auto layout =
        StridedLayoutAttr::get(op.getContext(), offset, staticStrides);
    Type elemType;
    if (op.isBlockPtr()) {
      elemType = getElementTypeBlockPtr(op);
    } else {
      elemType = getElementTypeStructuredPtr(op);
    }
    return MemRefType::get(resultShape, elemType, layout);
  }

  // If there are dimensions with size 1 and stride 0, replace 0 stride with
  // the product of sizes of all lower dimensions. This avoids creating memref
  // with zero stride.
  static llvm::SmallVector<OpFoldResult>
  getMixedStridesForMemref(tts::MakeTensorPtrOp op, OpBuilder &b) {
    llvm::SmallVector<OpFoldResult> strides;
    auto accumulate = 1;
    for (auto [size, stride] :
         llvm::reverse(llvm::zip(op.getSizes(), op.getMixedStrides()))) {
      auto strideIntAttr = getIntAttr(stride);
      if (size == 1 && strideIntAttr && strideIntAttr.value() == 0) {
        strides.push_back(b.getIndexAttr(accumulate));
      } else {
        strides.push_back(stride);
      }
      accumulate *= size;
    }
    std::reverse(strides.begin(), strides.end());
    return strides;
  }

  static OpFoldResult accumulateTargetOffset(tts::MakeTensorPtrOp op,
                                             OpBuilder &b) {
    Location loc = op->getLoc();
    OpFoldResult targetOffset = b.getIndexAttr(0);
    for (auto o : op.getMixedOffsets()) {
      targetOffset = addOFRs(targetOffset, o, loc, b);
    }
    return targetOffset;
  }

  std::pair<memref::ReinterpretCastOp, memref::ReinterpretCastOp>
  createSideBySideCastOps(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto resultShape = cast<RankedTensorType>(op.getType()).getShape();

    auto targetOffset =
        ofrToIndexValue(accumulateTargetOffset(op, rewriter), loc, rewriter);

    ////////////////////////////////////////////////////////////////////////////
    //
    // Handling side-by-side wraparound
    //
    // Note: We do not support cases where the target has already overflown the
    // number of columns! This is because in PtrAnalysis, the offset has already
    // been collapsed into a single dimension, so it is ambiguous to determine
    // whether the offset actually overflows or just refers to an element on the
    // subsequent rows.
    //
    // Same limitations apply to the stacked wraparound case.
    //
    ////////////////////////////////////////////////////////////////////////////
    //
    //    nextOffset - targetOffset = colSize
    //    d1 + d2 = colSize
    //                          N
    //                                x            clampedOffset
    //      --------------------------*----------------*-----*
    //      |                                          |     nextOffset (might
    //      |                    targetOffset          |             overflow)
    //  y   *-----                    *----------------|
    //      |    |                    |                |
    //  M   |-----                    -----------------|
    //      | d2                              d1       |
    //      --------------------------------------------
    //
    //    x = targetOffset % N
    //    nextOffset = x + colSize
    //    clampedOffset = min(nextOffset, N)
    //    d1 = clampedOffset - x
    //
    ////////////////////////////////////////////////////////////////////////////

    auto resultType = getResultMemrefType(
        op, /* offset */ ShapedType::kDynamic,
        /* staticStrides */
        SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic),
        /* result shape */
        SmallVector<int64_t>{

            // Row stays the same, but mlir doesn't allow this anymore. Put
            // dynamic.
            ShapedType::kDynamic,

            // Column is dynamic, in most cases, this
            // should be the same as the original column.
            // The last chunk may be smaller due to
            // wrapping around.
            ShapedType::kDynamic});

    Value rowSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[0]));
    Value colSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[1]));

    Value modN = ofrToIndexValue(op.getMixedShape()[1], loc, rewriter);

    Value x = rewriter.create<arith::RemSIOp>(loc, targetOffset, modN);
    Value y = rewriter.create<arith::SubIOp>(loc, targetOffset, x);

    SmallVector<Value> strideVals =
        ofrsToIndexValues(op.getMixedStrides(), loc, rewriter);

    // First chunk
    Value nextOffset = rewriter.create<arith::AddIOp>(loc, x, colSize);
    Value clampedOffset =
        rewriter.create<arith::MinSIOp>(loc, nextOffset, modN);
    Value d1 = rewriter.create<arith::SubIOp>(loc, clampedOffset, x);

    auto cast1 = rewriter.create<memref::ReinterpretCastOp>(
        loc, resultType, adaptor.getBase(), targetOffset,
        ValueRange{rowSize, d1}, strideVals);

    // Second chunk
    Value d2 = rewriter.create<arith::SubIOp>(loc, colSize, d1);

    auto cast2 = rewriter.create<memref::ReinterpretCastOp>(
        loc, resultType, adaptor.getBase(), y, ValueRange{rowSize, d2},
        strideVals);

    return {cast1, cast2};
  }

  std::pair<memref::ReinterpretCastOp, memref::ReinterpretCastOp>
  createStackedCastOps(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {

    auto loc = op->getLoc();
    auto resultShape = cast<RankedTensorType>(op.getType()).getShape();

    assert(resultShape.size() == 2);

    auto targetOffset =
        ofrToIndexValue(accumulateTargetOffset(op, rewriter), loc, rewriter);

    ////////////////////////////////////////////////////////////////////////////
    //
    // Handling stacked wraparound
    //
    // We do not support cases where the target offset has already overflown the
    // number of rows. See side-by-side wraparound for details.
    //
    ////////////////////////////////////////////////////////////////////////////
    //    We're loading a tensor of dim (rowSize, colSize)
    //    d1 + d2 = rowSize
    //    d2 is the number of rows that overflow
    //
    //                       cols
    //
    //               wrappedAroundOff
    //      --------------*------------*--------
    //      |        d2   |            |       |
    //      |             |------------|       |
    //  rows|                                  |
    //      |                                  |
    //      |           targetOffset           |
    //      |             *------------|       |
    //      |             |            |       |
    //      |         d1  |            |       |
    //      |             | clampedOff |       |
    //      --------------*---------------------
    //                    |  overflow  |
    //                    *-------------
    //                 nextOff
    //
    //    wrappedAroundOff = targetOffset % cols
    //    clampedOff = (rows * strideRows) + wrappedAroundOff
    //                  ~~~~~~~~~~~~~~~~~
    //                         ^
    //                         |
    //          We have already computed
    //          rows * strideRows = modRow = shape[1]
    //          in TritonToStructured
    //
    //          clampedOff - targetOffset
    //    d1 = --------------------
    //              strideRows

    auto resultType = getResultMemrefType(
        op, /* offset */ ShapedType::kDynamic,
        /* staticStrides */
        SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic),
        /* result shape */
        SmallVector<int64_t>{
            // Row is dynamic, in most cases, this should
            // be the same as the original row. The last
            // chunk may be smaller due to wrapping
            // around.
            ShapedType::kDynamic,

            // Col stays the same, which is resultShape[1], but mlir doesn't
            // allow this anymore. So we put dynamic instead.
            ShapedType::kDynamic});

    Value rowSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[0]));
    Value colSize = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(op.getSizes()[1]));

    Value strideRow = ofrToIndexValue(op.getMixedStrides()[0], loc, rewriter);
    Value strideCol = ofrToIndexValue(op.getMixedStrides()[1], loc, rewriter);

    Value modRow = op.getShape()[0];

    // First chunk
    Value wrappedAroundOff =
        rewriter.create<arith::RemSIOp>(loc, targetOffset, strideRow);
    Value clampedOff =
        rewriter.create<arith::AddIOp>(loc, modRow, wrappedAroundOff);
    Value d1 = rewriter.create<arith::SubIOp>(loc, clampedOff, targetOffset);
    d1 = rewriter.create<arith::DivSIOp>(loc, d1, strideRow);

    memref::ReinterpretCastOp cast1 =
        rewriter.create<memref::ReinterpretCastOp>(
            loc, resultType, adaptor.getBase(), targetOffset,
            ValueRange{d1, colSize}, ValueRange{strideRow, strideCol});

    // Second chunk
    Value d2 = rewriter.create<arith::SubIOp>(loc, rowSize, d1);
    memref::ReinterpretCastOp cast2 =
        rewriter.create<memref::ReinterpretCastOp>(
            loc, resultType, adaptor.getBase(), wrappedAroundOff,
            ValueRange{d2, colSize}, ValueRange{strideRow, strideCol});

    return {cast1, cast2};
  }

  LogicalResult rewriteSplitPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {

    auto parentShape = op.getStaticShape();

    SmallVector<Value> casts;
    StringRef wrapType;

    if (parentShape[0] == ShapedType::kDynamic) {
      // Stacked case
      assert(parentShape[1] == 0);
      auto [cast1, cast2] = createStackedCastOps(op, adaptor, rewriter);
      casts = {cast1.getResult(), cast2.getResult()};
      wrapType = WRAP_STACKED;
    } else {
      assert(parentShape[0] == 0);
      auto [cast1, cast2] = createSideBySideCastOps(op, adaptor, rewriter);
      casts = {cast1.getResult(), cast2.getResult()};
      wrapType = WRAP_SIDE_BY_SIDE;
    }

    auto combinedCast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), op.getType(), casts);

    combinedCast->setAttr(wrapType, rewriter.getUnitAttr());

    rewriter.replaceOp(op, combinedCast);

    return success();
  }

  LogicalResult rewritePtr(ArrayRef<int64_t> resultShape, bool isBlockPtr,
                           tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {

    auto mixedStrides = getMixedStridesForMemref(op, rewriter);
    SmallVector<int64_t> staticStrides;
    SmallVector<Value> dynamicStrides;
    dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

    auto targetOffset = accumulateTargetOffset(op, rewriter);
    auto staticTargetOffset = getIntAttr(targetOffset);
    auto resultType = getResultMemrefType(
        op, staticTargetOffset.value_or(ShapedType::kDynamic), staticStrides,
        resultShape);

    auto castOp = rewriter.create<memref::ReinterpretCastOp>(
        op.getLoc(), resultType, adaptor.getBase(), targetOffset,
        op.getMixedSizes(), mixedStrides);

    rewriter.replaceOp(op, castOp);

    return success();
  }

  LogicalResult
  rewriteStructuredPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    ArrayRef<int64_t> resultShape = cast<ShapedType>(op.getType()).getShape();
    return rewritePtr(resultShape, false, op, adaptor, rewriter);
  }

  LogicalResult rewriteBlockPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Block pointers are basically the same as structured pointers except that
    // the return types are !tt.ptr<tensor<AxBxCxbf16>> instead of
    // tensor<AxBxCx!tt.ptr<bf16>>
    ArrayRef<int64_t> resultShape =
        cast<ShapedType>(
            cast<triton::PointerType>(op.getType()).getPointeeType())
            .getShape();
    return rewritePtr(resultShape, true, op, adaptor, rewriter);
  }

public:
  MakeTensorPtrConverter(const TypeConverter &typeConverter,
                         MLIRContext *context)
      : OpConversionPattern<tts::MakeTensorPtrOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!llvm::is_sorted(op.getOrder(), std::greater<>())) {
      emitError(op.getLoc()) << "non-decreasing dimension order on tensor "
                                "pointers are not yet supported";
      return failure();
    }
    op.dump();
    if (op.isBlockPtr()) {
      return rewriteBlockPtr(op, adaptor, rewriter);
    }

    if (op.isStructuredPtr()) {
      return rewriteStructuredPtr(op, adaptor, rewriter);
    }

    if (op.isSplitPtr()) {
      return rewriteSplitPtr(op, adaptor, rewriter);
    }

    return failure();
  }
};

/// Handles conversion of Triton's structured LoadOp to Memref-based operations.
/// This includes support for both contiguous memory accesses and wrapped memory
/// patterns (side-by-side and stacked layouts).
struct LoadConverter : public OpConversionPattern<tts::LoadOp> {
private:
  using OpConversionPattern<tts::LoadOp>::OpConversionPattern;

  /// Creates a new tensor by copying two memory blocks in side-by-side layout.
  /// This handles cases where valid data wraps around column boundaries.
  ///
  /// @param block1 First memory block (left side)
  /// @param block2 Second memory block (right side)
  /// @param dst Destination tensor to write combined data
  /// @param loc Operation location for IR generation
  /// @param rewriter Pattern rewriter for IR modification
  /// @returns Combined tensor with data from both blocks
  Value createSideBySideCopies(Value block1, Value block2, Value dst,
                               Location loc,
                               ConversionPatternRewriter &rewriter) const {
    // Initialize constants for index operations
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Extract dimensions from memory blocks
    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    // Convert memory blocks to tensors for insertion
    Value tensor1 = rewriter.create<bufferization::ToTensorOp>(loc, block1);
    Value tensor2 = rewriter.create<bufferization::ToTensorOp>(loc, block2);

    Value inserted1 = rewriter.create<tensor::InsertSliceOp>(
        loc, tensor1, dst, ValueRange{zero, zero},
        ValueRange{block1Row, block1Col},
        ValueRange{one, one}); // Stride 1 for both dimensions

    // Insert second block at right half starting after first block's columns
    Value colOffset = rewriter.create<arith::AddIOp>(loc, zero, block1Col);
    Value inserted2 = rewriter.create<tensor::InsertSliceOp>(
        loc, tensor2, inserted1, ValueRange{zero, colOffset},
        ValueRange{block1Row, block2Col}, ValueRange{one, one});
    return inserted2;
  }

  /// Creates a new tensor by copying two memory blocks in stacked layout.
  /// This handles cases where valid data wraps around row boundaries.
  ///
  /// @param block1 First memory block (upper half)
  /// @param block2 Second memory block (lower half)
  /// @param dst Destination tensor to write combined data
  /// @param loc Operation location for IR generation
  /// @param rewriter Pattern rewriter for IR modification
  /// @returns Combined tensor with data from both blocks
  Value createStackedCopies(Value block1, Value block2, Value dst, Location loc,
                            ConversionPatternRewriter &rewriter) const {
    // Initialize constants for index operations
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Extract dimensions from memory blocks
    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);
    Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    // Convert memory blocks to tensors for insertion
    Value tensor1 = rewriter.create<bufferization::ToTensorOp>(loc, block1);
    Value tensor2 = rewriter.create<bufferization::ToTensorOp>(loc, block2);

    // Insert first block at upper half (row 0 to block1Row)
    Value inserted1 = rewriter.create<tensor::InsertSliceOp>(
        loc, tensor1, dst, ValueRange{zero, zero},
        ValueRange{block1Row, block1Col}, ValueRange{one, one});

    // Insert second block at lower half starting after first block's rows
    Value rowOffset = rewriter.create<arith::AddIOp>(loc, zero, block1Row);
    Value inserted2 = rewriter.create<tensor::InsertSliceOp>(
        loc, tensor2, inserted1, ValueRange{rowOffset, zero},
        ValueRange{block2Row, block2Col}, ValueRange{one, one});

    return inserted2;
  }

  /// Creates a subview operation for the specified memory region.
  ///
  /// @param src Source memref to create subview from
  /// @param offsets Starting indices for the subview
  /// @param sizes Dimensions of the subview
  /// @param loc Operation location for IR generation
  /// @param rewriter Pattern rewriter for IR modification
  /// @returns Memref subview operation
  memref::SubViewOp createSubview(Value src, ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes,
                                  ArrayRef<OpFoldResult> strides, Location loc,
                                  ConversionPatternRewriter &rewriter) const {
    auto srcType = cast<MemRefType>(src.getType());
    auto dstType =
        memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
    return rewriter.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType),
                                              src, offsets, sizes, strides);
  }

  /// Creates subviews for side-by-side wrapped memory case.
  /// @param dims Target dimensions for the final tensor
  /// @param block1 First memory block (left side)
  /// @param block2 Second memory block (right side)
  /// @returns Pair of subviews covering valid data regions
  std::pair<memref::SubViewOp, memref::SubViewOp>
  getSideBySideSubviews(ArrayRef<OpFoldResult> dims, Value block1, Value block2,
                        Location loc,
                        ConversionPatternRewriter &rewriter) const {
    OpFoldResult subviewRowFull = dims[0];
    OpFoldResult subviewColFull = dims[1];
    OpFoldResult col1 =
        rewriter.create<memref::DimOp>(loc, block1, 1).getResult();
    OpFoldResult subviewCol1 = minOFRs(col1, subviewColFull, loc, rewriter);
    OpFoldResult subviewCol2 =
        subOFRs(subviewColFull, subviewCol1, loc, rewriter);

    SmallVector<OpFoldResult> offsets(dims.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(dims.size(), rewriter.getIndexAttr(1));
    auto sv1 = createSubview(block1, offsets, {subviewRowFull, subviewCol1},
                             strides, loc, rewriter);
    auto sv2 = createSubview(block2, offsets, {subviewRowFull, subviewCol2},
                             strides, loc, rewriter);

    return {sv1, sv2};
  }

  /// Creates subviews for stacked wrapped memory case.
  /// @param dims Target dimensions for the final tensor
  /// @param block1 First memory block (upper half)
  /// @param block2 Second memory block (lower half)
  /// @returns Pair of subviews covering valid data regions
  std::pair<memref::SubViewOp, memref::SubViewOp>
  getStackedSubviews(ArrayRef<OpFoldResult> dims, Value block1, Value block2,
                     Location loc, ConversionPatternRewriter &rewriter) const {
    OpFoldResult subviewRowFull = dims[0];
    OpFoldResult subviewColFull = dims[1];
    OpFoldResult row1 =
        rewriter.create<memref::DimOp>(loc, block1, 0).getResult();
    OpFoldResult subviewRow1 = minOFRs(row1, subviewRowFull, loc, rewriter);
    OpFoldResult subviewRow2 =
        subOFRs(subviewRowFull, subviewRow1, loc, rewriter);

    SmallVector<OpFoldResult> offsets(dims.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(dims.size(), rewriter.getIndexAttr(1));
    auto sv1 = createSubview(block1, offsets, {subviewRow1, subviewColFull},
                             strides, loc, rewriter);
    auto sv2 = createSubview(block2, offsets, {subviewRow2, subviewColFull},
                             strides, loc, rewriter);
    return {sv1, sv2};
  }

  /// Handles conversion for structured loads without mask.
  LogicalResult
  rewriteStructuredLoad(tts::LoadOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    MLIRContext *context = op->getContext();
    auto loc = op->getLoc();
    Value ptr = adaptor.getPtr();
    auto tensorType = cast<RankedTensorType>(op.getType());

    // Create empty tensor initialized with zeros
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        loc, tensorType.getShape(), tensorType.getElementType());
    Value zeroVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(tensorType.getElementType()));
    auto filledTensor =
        rewriter.create<linalg::FillOp>(loc, zeroVal, emptyOp.getResult())
            .getResult(0);
    filledTensor.getDefiningOp()->setAttr(
        "triton_ptr", mlir::tts::TritonPtrAttr::get(context));

    // Handle wrapped memory cases
    if (auto castOp = ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
      auto memrefs = castOp.getOperands();
      assert(memrefs.size() == 2 && "Expected two memrefs for wrapped case");

      if (castOp->hasAttr(WRAP_SIDE_BY_SIDE)) {
        Value result = createSideBySideCopies(memrefs[0], memrefs[1], emptyOp,
                                              loc, rewriter);
        rewriter.replaceOp(op, result);
      } else if (castOp->hasAttr(WRAP_STACKED)) {
        Value result =
            createStackedCopies(memrefs[0], memrefs[1], emptyOp, loc, rewriter);
        rewriter.replaceOp(op, result);
      }
    } else {
      // Simple contiguous memory case
      Value srcTensor =
          rewriter.create<bufferization::ToTensorOp>(loc, ptr, true, false);

      SmallVector<OpFoldResult> sizes;
      for (auto dim : tensorType.getShape()) {
        sizes.push_back(rewriter.getIndexAttr(dim));
      }
      Value inserted = rewriter.create<tensor::InsertSliceOp>(
          loc, srcTensor, filledTensor,
          SmallVector<OpFoldResult>(tensorType.getRank(),
                                    rewriter.getIndexAttr(0)),
          sizes, // 使用转换后的sizes
          SmallVector<OpFoldResult>(tensorType.getRank(),
                                    rewriter.getIndexAttr(1)));

      rewriter.replaceOp(op, inserted);
    }
    return success();
  }

  /// Handles conversion for masked loads with boundary checks.
  LogicalResult rewriteMaskedLoad(tts::LoadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    MLIRContext *context = op->getContext();
    auto loc = op->getLoc();
    Value ptr = adaptor.getPtr();
    auto tensorType = cast<RankedTensorType>(op.getType());

    // Create padded tensor initialized with out-of-bound values
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        loc, tensorType.getShape(), tensorType.getElementType());
    Value paddingVal =
        op.getOther()
            ? op.getOther()
            : rewriter.create<arith::ConstantOp>(
                  loc, rewriter.getZeroAttr(tensorType.getElementType()));
    Value emptyTensor = emptyOp.getResult();

    Value paddedTensor =
        rewriter.create<linalg::FillOp>(loc, paddingVal, emptyTensor)
            .getResult(0);
    paddedTensor.getDefiningOp()->setAttr(
        "triton_ptr", mlir::tts::TritonPtrAttr::get(context));

    // Handle wrapped memory cases
    if (auto castOp = ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
      auto memrefs = castOp.getOperands();
      assert(memrefs.size() == 2 && "Expected two memrefs for wrapped case");
      auto mixedDims = op.getMixedMaskDims();

      // Create valid-range subviews and insert into padded tensor
      if (castOp->hasAttr(WRAP_SIDE_BY_SIDE)) {
        auto [sv1, sv2] = getSideBySideSubviews(mixedDims, memrefs[0],
                                                memrefs[1], loc, rewriter);
        Value result =
            createSideBySideCopies(sv1, sv2, paddedTensor, loc, rewriter);
        rewriter.replaceOp(op, result);
      } else if (castOp->hasAttr(WRAP_STACKED)) {
        auto [sv1, sv2] = getStackedSubviews(mixedDims, memrefs[0], memrefs[1],
                                             loc, rewriter);
        Value result =
            createStackedCopies(sv1, sv2, paddedTensor, loc, rewriter);
        rewriter.replaceOp(op, result);
      }
      rewriter.eraseOp(castOp);
    } else {
      // Simple masked load with single memref
      auto mixedDims = op.getMixedMaskDims();
      SmallVector<OpFoldResult> strides(mixedDims.size(),
                                        rewriter.getIndexAttr(1));
      memref::SubViewOp srcSubview =
          createSubview(ptr,
                        SmallVector<OpFoldResult>(tensorType.getRank(),
                                                  rewriter.getIndexAttr(0)),
                        mixedDims, strides, loc, rewriter);
      Value srcTensor = rewriter.create<bufferization::ToTensorOp>(
          loc, srcSubview, true, false);
      Value inserted = rewriter.create<tensor::InsertSliceOp>(
          loc, srcTensor, paddedTensor,
          SmallVector<OpFoldResult>(tensorType.getRank(),
                                    rewriter.getIndexAttr(0)),
          mixedDims,
          SmallVector<OpFoldResult>(tensorType.getRank(),
                                    rewriter.getIndexAttr(1)));
      rewriter.replaceOp(op, inserted);
    }
    return success();
  }

public:
  LoadConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::LoadOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return op.hasMask() ? rewriteMaskedLoad(op, adaptor, rewriter)
                        : rewriteStructuredLoad(op, adaptor, rewriter);
  }
};

struct StoreConverter : public OpConversionPattern<tts::StoreOp> {
private:
  using OpConversionPattern<tts::StoreOp>::OpConversionPattern;

  static tensor::ExtractSliceOp
  getExtractSlice(int rank, ArrayRef<OpFoldResult> dims, Value source,
                  const Location loc, OpBuilder &b) {
    auto sourceType = cast<RankedTensorType>(source.getType());
    SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));

    auto dstType = tensor::ExtractSliceOp::inferResultType(sourceType, offsets,
                                                           dims, strides);

    return b.create<tensor::ExtractSliceOp>(loc, dstType, source, offsets, dims,
                                            strides);
  }

public:
  StoreConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::StoreOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ptr = adaptor.getPtr();
    auto storeValue = op.getValue();
    auto rank = cast<RankedTensorType>(storeValue.getType()).getRank();

    if (op.hasMask()) {
      auto mixedDims = op.getMixedMaskDims();

      auto srcSlice =
          getExtractSlice(rank, mixedDims, storeValue, loc, rewriter);
      auto dstSubview = getSubview(rank, mixedDims, ptr, loc, rewriter);

      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, srcSlice, dstSubview);
      storeOp.setWritable(true);
    } else {
      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, storeValue, ptr);
      storeOp.setWritable(true);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::populateStructuredToMemrefConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<MakeTensorPtrConverter>(typeConverter, patterns.getContext());
  patterns.add<LoadConverter, StoreConverter>(patterns.getContext());
}
