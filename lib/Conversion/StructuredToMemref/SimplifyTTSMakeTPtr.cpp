#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

using namespace mlir;

namespace {

struct AnalyzeAndTransformTTSMakeTPtrPattern
    : public OpConversionPattern<tts::MakeTensorPtrOp> {
  using OpConversionPattern<tts::MakeTensorPtrOp>::OpConversionPattern;

  AnalyzeAndTransformTTSMakeTPtrPattern(const TypeConverter &typeConverter,
                         MLIRContext *context)
      : OpConversionPattern<tts::MakeTensorPtrOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter)  const override {
    op.emitRemark() << "AnalyzeAndTransformTTSMakeTPtrPattern  applied successfully";
    bool changed = false;
    SmallVector<OpFoldResult, 4> shapeMixed = op.getMixedShape();

    // Process each dimension of the shape.
    for (unsigned d = 0, e = shapeMixed.size(); d < e; ++d) {
      if (analyzeMakeTPtrDimension(op, d)) {
        transformMakeTPtrDimension(op, d, rewriter);
        changed = true;
      }
    }
    if (changed)
      op.emitRemark() << "Transformed static_shape dimension(s) to 0 based on "
                         "use-def analysis";
    return success();
  }

private:
  /// This helper function analyzes the use-def chain for a given mask value.
  /// It verifies that the chain follows the expected pattern:
  ///   mask = arith.minsi( arith.subi( offset, arith.maxsi( offset,
  ///            arith.minsi(shapeConst, ...) ) ), tensorDim )
  /// and that the provided 'expectedBlockDimSize' matches the size from
  /// the corresponding tts.make_tptr. Returns true if the chain matches.
  ///
  /// Explanation:
  /// The mask computation usually ensures that the tiled dimension does not
  /// exceed the tensor dimension. For example, given:
  ///   offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
  ///   c_ptrs = c_ptr + stride_cn * offs_cn[None, :]
  ///   tl.store(c_ptrs, c, mask = offs_cn[None, :] < N)
  /// the condition `offs_cn[None, :] < N` is used to compute the mask. In the
  /// mask analysis, this condition is transformed into a series of operations:
  ///   %newEnd   = arith.min %lhsEnd, N
  ///   %newEnd2  = arith.max %newEnd, (pid_n * BLOCK_SIZE_N)
  ///   %newDim   = arith.sub %newEnd2, (pid_n * BLOCK_SIZE_N)
  ///
  /// If the stored variable's corresponding tts.make_tptr has a shape equal to
  /// N and its offset is also `pid_n * BLOCK_SIZE_N`, then the shape element N
  /// can be set to 0 (since the portion beyond N is not used and is redundant).
  /// For safety, we only allow this transformation if the variable is not
  /// modified elsewhere (i.e. its direct and indirect uses are each exactly
  /// one), following the strict usage path: tts.make_tptr -> tts.load ->
  /// (linalg op) -> tts.store. In particular, the result of tts.load must only
  /// be used as an input operand to a linalg op, ensuring that the original
  /// value remains unmodified.
  bool analyzeMaskChain(Value maskVal, Value expectedOffset,
                        int64_t makeTPtrShape,
                        int64_t expectedBlockDimSize) const {

    // When computing the mask, we need to ensure that the tile size does not
    // exceed the tensor dimension. For example, in the case:
    //   offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    //   c_ptrs = c_ptr + stride_cn * offs_cn[None, :]
    //   tl.store(c_ptrs, c, mask = offs_cn[None, :] < N)
    // the condition `offs_cn[None, :] < N` is the mask computation step.
    // In the mask analysis, this is transformed into:
    //   %newEnd   = arith.min %lhsEnd, N
    //   %newEnd2  = arith.max %newEnd, (pid_n * BLOCK_SIZE_N)
    //   %newDim   = arith.sub %newEnd2, (pid_n * BLOCK_SIZE_N)
    //
    // If the corresponding tts.make_tptr variable being stored has a shape
    // value N and its offset is also `pid_n * BLOCK_SIZE_N`, then we can change
    // the shape element from N to 0. This transformation is only allowed if the
    // stored variable is not used or modified elsewhere, and both its direct
    // and indirect uses are single-use.

    Value newDim = maskVal;
    // In cases where there are multiple mask computations (e.g.,
    // "c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)" in tl.store),
    // ensure that the computed new dimension does not exceed the corresponding
    // tensor dimension.
    auto minsiOp = maskVal.getDefiningOp<arith::MinSIOp>();
    if (minsiOp) {
      bool foundTensorAxis = false;
      for (Value operand : minsiOp.getOperands()) {
        if (auto constOp = operand.getDefiningOp<arith::ConstantOp>()) {
          if (mlir::cast<IntegerAttr>(constOp.getValue()).getInt() ==
              expectedBlockDimSize) {
            foundTensorAxis = true;
            // Select the operand that is not the constant (i.e. the dynamic
            // value).
            newDim =
                minsiOp.getOperand(operand == minsiOp.getOperand(0) ? 1 : 0);
          }
        }
      }
      if (!foundTensorAxis)
        return false;
    }

    // Look for a subtraction operation in the chain.
    auto subiOp = newDim.getDefiningOp<arith::SubIOp>();
    if (!subiOp)
      return false;

    // Check that the final statement in the transformed mask computation
    // (i.e., "%newDim = arith.sub %newEnd2, (pid_n * BLOCK_SIZE_N)") has the
    // expected offset.
    bool hasExpectedOffset = false;
    Value newEndInSub = nullptr;
    for (Value operand : subiOp.getOperands()) {
      if (operand == expectedOffset)
        hasExpectedOffset = true;
      else
        newEndInSub = operand;
    }
    if (!hasExpectedOffset || !newEndInSub)
      return false;

    // Next, verify that the previous operation in the chain is a max operation:
    // "%newEnd2 = arith.max %newEnd, (pid_n * BLOCK_SIZE_N)".
    auto maxsiOp = newEndInSub.getDefiningOp<arith::MaxSIOp>();
    if (!maxsiOp)
      return false;
    // One operand of the max must be the expected offset.
    bool foundOffsetInMax = false;
    Value newEnd = nullptr;
    for (Value operand : maxsiOp.getOperands()) {
      if (operand == expectedOffset)
        foundOffsetInMax = true;
      else
        newEnd = operand;
    }
    if (!foundOffsetInMax || !newEnd)
      return false;

    // Finally, ensure that the other operand of the max originates from a min
    // operation that uses a constant equal to the make_tptr shape value.
    auto innerMinsiOp = newEnd.getDefiningOp<arith::MinSIOp>();
    if (!innerMinsiOp)
      return false;
    bool foundShapeConst = false;
    for (Value operand : innerMinsiOp.getOperands()) {
      if (auto constOp = operand.getDefiningOp<arith::ConstantOp>()) {
        if (mlir::cast<IntegerAttr>(constOp.getValue()).getInt() ==
            makeTPtrShape)
          foundShapeConst = true;
      }
    }
    if (!foundShapeConst)
      return false;

    return true;
  }

  /// Checks whether the given linalg op uses the tts.load result only as an
  /// input. Returns true if at least one operand (with operand number <
  /// numDpsInputs) matches.
  bool isValidLinalgUser(Operation *linalgOp, tts::LoadOp *loadOp) const {
    auto linalg = cast<linalg::LinalgOp>(linalgOp);
    for (OpOperand &operand : linalgOp->getOpOperands()) {
      if (operand.get() == loadOp->getResult()) {
        // Check that the operand is used as a data (input) operand.
        if (operand.getOperandNumber() < linalg.getNumDpsInputs())
          return true;
      }
    }
    return false;
  }

  /// Helper function that analyzes the use–def chain for a given dimension
  /// of a tts.make_tptr op. It checks if:
  ///   - The shape element (from the static_shape attribute) is nonzero,
  ///   - The op's result has exactly one use, and that use is a tts.load op,
  ///   - There exists an indirect use in a tts.store op where the corresponding
  ///   mask
  ///     index is computed through a chain of arith.min, arith.sub, arith.max,
  ///     and an inner arith.min operation that uses the offset and the shape
  ///     value,
  ///   - The size from getMixedSizes matches the expected tensor dimension.
  /// Returns true if the dimension satisfies the pattern.
  bool analyzeMakeTPtrDimension(tts::MakeTensorPtrOp op, unsigned d) const {
    // Retrieve the mixed shape list and check the d-th element.
    SmallVector<OpFoldResult, 4> shapeMixed = op.getMixedShape();
    if (d >= shapeMixed.size())
      return false;
    Attribute attr = shapeMixed[d].dyn_cast<Attribute>();
    if (!attr)
      return false;
    auto shapeAttr = mlir::dyn_cast<IntegerAttr>(attr);
    if (!shapeAttr)
      return false;

    int64_t shapeVal = shapeAttr.getInt();
    if (shapeVal == 0)
      return false; // Already zero; nothing to transform.

    // Check that the op result has exactly one use.
    Value opResult = op.getResult();
    if (!opResult.hasOneUse())
      return false;

    // Ensure that the direct use is a tts.load op.
    Operation *ttsLoadOp = *opResult.user_begin();
    if (!isa<tts::LoadOp>(ttsLoadOp) || !ttsLoadOp->hasOneUse())
      return false;

    // Search for an indirect use that eventually leads to a tts.store op.
    Value storeMaskCandidate;
    Operation *loadUser = *(ttsLoadOp->user_begin());
    if (!loadUser->hasOneUse())
      return false;
    // Check if the user operation is from the linalg dialect or a tts.store op.
    if (loadUser->getDialect()->getNamespace() == "linalg") {
      // For linalg ops, verify that the tts.load result is used only as an
      // input operand.
      bool isInput = false;
      for (OpOperand &operand : loadUser->getOpOperands()) {
        if (operand.get() == ttsLoadOp->getResult(0)) {
          // Verify that this operand belongs to the input parameters.
          if (operand.getOperandNumber() <
              cast<linalg::LinalgOp>(loadUser).getNumDpsInputs()) {
            isInput = true;
            break;
          }
        }
      }
      if (!isInput) {
        return false;
      }
      auto ttsStoreOp = *(loadUser->user_begin());
      if (ttsStoreOp == nullptr || !isa<tts::StoreOp>(ttsStoreOp) ||
          !ttsStoreOp->hasOneUse())
        return false;

      auto storeOp = cast<tts::StoreOp>(ttsStoreOp);

      // For a tts.store op, check that the tts.load result is used as the
      // second operand.
      if (storeOp.getOperand(1) != ttsLoadOp->getResult(0)) {
        return false;
      }

      unsigned maskIdx = 2 + d;
      if (storeOp.getNumOperands() > maskIdx) {
        storeMaskCandidate = storeOp.getOperand(maskIdx);
      }

    } else if (auto storeOp = dyn_cast<tts::StoreOp>(loadUser)) {
      // For a tts.store op, check that the tts.load result is used as the
      // second operand.
      if (storeOp.getOperand(1) != ttsLoadOp->getResult(0)) {
        return false;
      }
      // Assume that the mask for dimension 'd' is at operand index (2 + d).
      unsigned maskIdx = 2 + d;
      if (storeOp.getNumOperands() > maskIdx) {
        storeMaskCandidate = storeOp.getOperand(maskIdx);
      }
    }

    if (!storeMaskCandidate)
      return false;

    // Retrieve the offset for dimension d.
    SmallVector<OpFoldResult, 4> offsetsMixed = op.getMixedOffsets();
    if (d >= offsetsMixed.size())
      return false;
    Value offsetVal = offsetsMixed[d].dyn_cast<Value>();
    if (!offsetVal)
      return false;

    // Retrieve the expected size from getMixedSizes.
    SmallVector<OpFoldResult, 4> sizesMixed = op.getMixedSizes();
    if (d >= sizesMixed.size())
      return false;

    Attribute sizeAttrOpFoldResult = sizesMixed[d].dyn_cast<Attribute>();
    if (!sizeAttrOpFoldResult)
      return false;
    auto sizeAttr = mlir::dyn_cast<IntegerAttr>(sizeAttrOpFoldResult);
    if (!sizeAttr)
      return false;

    int64_t expectedBlockDimSize = sizeAttr.getInt();

    // Call the helper function to analyze the mask chain.
    // (See the definition of analyzeMaskChain above for details.)
    return analyzeMaskChain(storeMaskCandidate, offsetVal, shapeVal,
                            expectedBlockDimSize);
  }

  /// Transforms a specific dimension of a tts.make_tptr operation by creating a
  /// new operation with adjusted static_shape attribute. The original dimension
  /// value in static_shape is replaced with 0, and all other parameters are
  /// kept identical. The original op is then replaced with the new one.
  void transformMakeTPtrDimension(tts::MakeTensorPtrOp op, unsigned d,
                                  PatternRewriter &rewriter) const {
    // Get current static_shape attribute
    auto staticShape = op.getStaticShape();
    SmallVector<int64_t> newShape(staticShape.begin(), staticShape.end());

    // Set target dimension to 0 for structured pointer case
    newShape[d] = 0;
    auto newStaticShape = rewriter.getDenseI64ArrayAttr(newShape);

    // Create new make_tptr op with identical parameters except modified shape
    auto newOp = rewriter.create<tts::MakeTensorPtrOp>(
        op.getLoc(), op.getResult().getType(),
        op.getBase(),          // TT_Ptr $base
        op.getSizes(),         // DenseI64ArrayAttr $sizes
        op.getStrides(),       // Variadic<Index> $strides
        op.getOffsets(),       // Variadic<Index> $offsets
        op.getShape(),         // Variadic<Index> $shape
        op.getStaticStrides(), // DenseI64ArrayAttr $static_strides
        op.getStaticOffsets(), // DenseI64ArrayAttr $static_offsets
        newStaticShape,        // Modified DenseI64ArrayAttr $static_shape
        op.getOrder()          // DenseI32ArrayAttr $order
    );

    // Replace original op with new version
    rewriter.replaceOp(op, newOp->getResults());
  }
};

} // end anonymous namespace
/// Helper function to add the rewrite pattern to a pattern list.
void mlir::triton::populateSimplifyTTSMakeTPtrPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<AnalyzeAndTransformTTSMakeTPtrPattern>(patterns.getContext());
}
