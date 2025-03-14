#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR//MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

// Structure to hold the dimension comparison parameters.
struct DimComparisonParams {
  // The dimension size from op.getMixedSizes() (possibly traced to a for-loop
  // parameter).
  Value dimShapeSize;
  // The dimension offset from op.getMixedOffsets() (or a derived value from
  // an arith.mul op).
  Value dimOffset;
  // The expected block dimension size from the op's result tensor type.
  int64_t blockDimSize;
};

struct AnalyzeAndTransformTTSMakeTPtrPattern
    : public OpRewritePattern<tts::MakeTensorPtrOp> {
  using OpRewritePattern<tts::MakeTensorPtrOp>::OpRewritePattern;

  AnalyzeAndTransformTTSMakeTPtrPattern(MLIRContext *context)
      : OpRewritePattern<tts::MakeTensorPtrOp>(context) {}

  LogicalResult matchAndRewrite(tts::MakeTensorPtrOp op,
                                PatternRewriter &rewriter) const {

    auto staticShape = op.getStaticShape();
    bool alreadyTransformed =
        llvm::all_of(staticShape, [](int64_t s) { return s == 0; });
    if (alreadyTransformed)
      return failure();

    bool changed = false;
    SmallVector<OpFoldResult, 4> shapeMixed = op.getMixedShape();
    SmallVector<int64_t> newShape(staticShape.begin(), staticShape.end());
    // Process each dimension of the shape.
    for (unsigned d = 0, e = shapeMixed.size(); d < e; ++d) {
      if (analyzeMakeTPtrDimension(op, d)) {
        newShape[d] = 0; // 将该维度置 0
        changed = true;
      }
    }
    // 如果有修改，则创建新的 op 并替换原 op
    if (changed) {
      auto newStaticShape = rewriter.getDenseI64ArrayAttr(newShape);
      // 使用 replaceOpWithNew 一步完成创建+替换
      rewriter.replaceOpWithNewOp<tts::MakeTensorPtrOp>(
          op,                       // 被替换的旧操作
          op.getResult().getType(), // 保持结果类型一致
          op.getBase(),             // 原参数
          op.getSizes(), op.getStrides(), op.getOffsets(), op.getShape(),
          op.getStaticStrides(), op.getStaticOffsets(),
          newStaticShape, // 修改后的新形状
          op.getOrder());
      return success();
    }
    return success();
  }

private:
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

  // Helper function: Check if it's an elementwise operation
  bool isElementwiseLinalg(linalg::LinalgOp op) const {
    return !op.hasIndexSemantics() &&
           op.getNumParallelLoops() == op.getNumLoops();
  }

  // Helper function: Check if it's a matmul operation (not just a general
  // contraction)
  bool isMatmulLinalg(linalg::LinalgOp op) const {
    // Exact match via operation name
    bool isMatmul = op->getName().getStringRef() == "linalg.matmul";
    if (!isMatmul)
      return false;
    // 第二步：验证维度特性
    auto verify2DShape = [](Value operand) -> bool {
      if (auto shapedType = mlir::dyn_cast<ShapedType>(operand.getType())) {
        return shapedType.hasRank() && shapedType.getRank() == 2;
      }
      return false;
    };

    // 检查所有输入操作数（A和B）
    for (Value input : op.getDpsInputs()) {
      if (!verify2DShape(input)) {
        return false;
      }
    }

    // 检查所有输出操作数（C）
    for (Value output : op.getDpsInits()) {
      if (!verify2DShape(output)) {
        return false;
      }
    }
    return true;
  }

  // Helper function: Verify single operation in the chain
  LogicalResult verifyLinalgOperation(Operation *linalgOp, Value sourceValue,
                                      unsigned makeTPtrShapeDim) const {
    auto cuurentLinalgOp = cast<linalg::LinalgOp>(linalgOp);

    // Check operation type,只允许elementwise操作和matmul操作
    const bool isValidType =
        isElementwiseLinalg(cuurentLinalgOp) || isMatmulLinalg(cuurentLinalgOp);
    if (!isValidType) {
      return failure();
    }

    // Check single-user constraint
    if (!linalgOp->hasOneUse()) {
      return failure();
    }

    // sourceValue只能作为linalg的输入参数
    for (auto i = 0; i < cuurentLinalgOp.getNumDpsInputs(); i++) {
      if (cuurentLinalgOp.getDpsInputs()[i] != sourceValue) {
        continue;
      }
      // Must be used as input operand
      if (i >= cuurentLinalgOp.getNumDpsInputs()) {
        return failure();
      }

      // 如果是matmul操作，则需要保证 makeTPtrShapeDim 不是reduce的维度
      if (isMatmulLinalg(cuurentLinalgOp)) {
        if (cuurentLinalgOp.getDpsInputs()[makeTPtrShapeDim] != sourceValue) {
          return failure();
        }
      }
    }
    return success();
  }

  // Trace use chain until reaching store operation，
  // tts.load -> (linalg.matmul / linalg.elemwise)* ->tts.store
  LogicalResult traceUseChainToStore(Value ttsLoadOpResult,
                                     unsigned makeTPtrShapeDim,
                                     Value &storeMaskCandidate) const {
    // Start with the result of the tts.load op.
    Value sourceValue = ttsLoadOpResult;
    bool seenMatmul =
        false; // Flag to ensure that a matmul op appears at most once.

    // --- Process the first chain of Linalg operations ---
    // This loop traverses the chain of Linalg operations (which can be
    // elementwise or matmul) that are directly using the source value from
    // tts.load.
    while (isa<linalg::LinalgOp>(*sourceValue.user_begin())) {
      // Ensure that the source value has exactly one user to maintain a strict
      // use-def chain.
      if (!sourceValue.hasOneUse())
        return failure();

      // Get the only user operation.
      Operation *currentUser = *sourceValue.user_begin();
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(currentUser)) {
        // If the op is a matmul, check that we haven't seen one before.
        if (isMatmulLinalg(linalgOp)) {
          if (seenMatmul)
            return failure(); // More than one matmul op is not allowed.
          seenMatmul = true;
        }
        // For non-matmul ops, ensure they are elementwise.
        else if (!isElementwiseLinalg(linalgOp)) {
          return failure(); // Unsupported linalg op type encountered.
        }

        // Verify that the current linalg op uses sourceValue appropriately.
        if (failed(verifyLinalgOperation(currentUser, sourceValue,
                                         makeTPtrShapeDim))) {
          return failure();
        }

        // Advance to the next value in the chain.
        sourceValue = currentUser->getResult(0);
        continue;
      }
      return failure(); // Unexpected op type encountered.
    }

    // --- Handle loop yield cases ---
    // The value might be yielded inside an scf.for loop.
    if (auto yieldOp = dyn_cast<scf::YieldOp>(*sourceValue.user_begin())) {
      // Get the parent op which should be the body of an scf.for loop.
      Operation *parentOp = yieldOp->getParentOp();
      auto forOp = dyn_cast<scf::ForOp>(parentOp);
      if (!forOp)
        return failure();

      // Identify which yield operand matches sourceValue.
      unsigned yieldIdx = 0;
      bool foundYield = false;
      for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
        if (yieldOp.getOperand(i) == sourceValue) {
          yieldIdx = i;
          foundYield = true;
          break;
        }
      }
      if (!foundYield)
        return failure();

      // Update sourceValue to the corresponding result from the scf.for op.
      sourceValue = forOp.getResult(yieldIdx);
    }

    // --- Handle direct scf.for iter_operand usage ---
    // Sometimes sourceValue is directly used as an iteration variable in an
    // scf.for op.
    if (auto forOp = dyn_cast<scf::ForOp>(*sourceValue.user_begin())) {
      bool foundIter = false;
      for (unsigned i = 0, n = forOp.getNumResults(); i < n; ++i) {
        if (forOp.getResult(i) == sourceValue) {
          sourceValue = forOp.getResult(i);
          foundIter = true;
          break;
        }
      }
      if (!foundIter)
        return failure();
    }

    // --- Process the second chain of Linalg operations, if present ---
    // There may be additional Linalg ops after handling scf constructs.
    while (isa<linalg::LinalgOp>(*sourceValue.user_begin())) {
      if (!sourceValue.hasOneUse())
        return failure();

      Operation *currentUser = *sourceValue.user_begin();
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(currentUser)) {
        // Again, ensure that matmul appears at most once.
        if (isMatmulLinalg(linalgOp)) {
          if (seenMatmul)
            return failure();
          seenMatmul = true;
        }
        // Verify that non-matmul ops are elementwise.
        else if (!isElementwiseLinalg(linalgOp)) {
          return failure();
        }

        // Verify that this linalg op properly uses the current sourceValue.
        if (failed(verifyLinalgOperation(currentUser, sourceValue,
                                         makeTPtrShapeDim))) {
          return failure();
        }
        // Continue along the chain.
        sourceValue = currentUser->getResult(0);
        continue;
      }
      return failure();
    }

    // --- Termination: Encountering the tts.store op ---
    // The use chain should eventually terminate with a tts.store op.
    if (auto storeOp = dyn_cast<tts::StoreOp>(*sourceValue.user_begin())) {
      // Verify that the tts.store op's second operand is our current
      // sourceValue.
      if (storeOp.getOperand(1) != sourceValue)
        return failure();

      // Compute the expected index for the mask operand based on
      // makeTPtrShapeDim.
      unsigned maskIdx = 2 + makeTPtrShapeDim;
      if (storeOp.getNumOperands() > maskIdx)
        storeMaskCandidate = storeOp.getOperand(maskIdx);

      // Return success if a valid mask candidate was identified.
      return success(storeMaskCandidate != nullptr);
    }

    // If none of the above conditions hold, then the use chain does not match
    // the expected pattern.
    return failure();
  }

  // Helper: If the value comes from a for-loop parameter, trace one level up to
  // trace one level up to obtain the original value.
  Value traceForLoopParameter(Value val) const {
    if (auto blockArg = mlir::dyn_cast<BlockArgument>(val)) {
      // 确认该参数属于 scf.for 的迭代参数
      if (auto forOp =
              dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
        // 获取迭代参数的索引（通常迭代参数在块参数中的起始位置为1，因为第一个参数是索引）
        unsigned iterArgIndex =
            blockArg.getArgNumber() - 1; // 调整索引，若循环有其他参数
                                         // 获取对应的初始值
        return forOp.getInitArgs()[iterArgIndex];
      }
    }
    return val;
  }

  // Encapsulated helper function to extract the three parameters for dimension
  // 'd'.
  std::optional<DimComparisonParams>
  getDimComparisonParams(tts::MakeTensorPtrOp &op, unsigned d) const {
    DimComparisonParams params;

    // Get the mixed offsets, sizes, and shapes vectors.
    auto offsetsMixed = op.getMixedOffsets();
    auto sizesMixed = op.getMixedSizes();
    auto shapeMixed = op.getMixedShape();

    // Ensure d is within range.
    if (d >= offsetsMixed.size() || d >= sizesMixed.size() ||
        d >= shapeMixed.size() || shapeMixed.size() > 2)
      return std::nullopt;

    // --- Step 3: Obtain the BlockDimSize from the op's result tensor type.
    // Note: We now do this after processing the offset.
    auto resultType =
        mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType || d >= resultType.getRank())
      return std::nullopt;
    params.blockDimSize = resultType.getDimSize(d);

    // --- Step 2: Decide how to extract dimShapeSize and dimOffset based on
    // offset.
    // 当makeTPtrOp的shape的维度为1，也就是低维度，不需要strid，可以直接获取对应的offset和具体的shape
    if (offsetsMixed.size() - d == 1) {

      Value shapeVal = shapeMixed[d].dyn_cast<Value>();
      if (!shapeVal)
        return std::nullopt;
      params.dimShapeSize = traceForLoopParameter(shapeVal);

      Value offsetVal = offsetsMixed[d].dyn_cast<Value>();
      if (!offsetVal)
        return std::nullopt;
      params.dimOffset = traceForLoopParameter(offsetVal);
    } else if (offsetsMixed.size() - d == 2) {
      // Case 2: When the offset is constant 2,
      // it indicates that the dimension is not the lowest and that the size and
      // offset are computed via a multiplication (arith.mul op). In this case,
      // we look for a common operand and then select the other operand from
      // each multiplication.
      // 高维度的计算，需要将每个维度的offset和shape乘以对应的stride才是实际的offset和shape
      Value shapeVal = shapeMixed[d].dyn_cast<Value>();
      Value offsetVal = offsetsMixed[d].dyn_cast<Value>();
      if (!shapeVal || !offsetVal)
        return std::nullopt;
      shapeVal = traceForLoopParameter(shapeVal);
      offsetVal = traceForLoopParameter(offsetVal);
      auto mulOpSize = shapeVal.getDefiningOp<arith::MulIOp>();
      auto mulOpOffset = offsetVal.getDefiningOp<arith::MulIOp>();
      if (!mulOpSize || !mulOpOffset)
        return std::nullopt;

      // Find the common operand between the two multiplication ops.
      Value strideOperand = nullptr;
      for (Value op1 : mulOpSize.getOperands()) {
        for (Value op2 : mulOpOffset.getOperands()) {
          if (op1 == op2) {
            strideOperand = op1;
            break;
          }
        }
        if (strideOperand)
          break;
      }
      if (!strideOperand)
        return std::nullopt;

      // For each multiplication op, select the operand that is not the common
      // operand.
      Value otherOperandSize = nullptr;
      for (Value opnd : mulOpSize.getOperands()) {
        if (opnd != strideOperand) {
          otherOperandSize = opnd;
          break;
        }
      }
      Value otherOperandOffset = nullptr;
      for (Value opnd : mulOpOffset.getOperands()) {
        if (opnd != strideOperand) {
          otherOperandOffset = opnd;
          break;
        }
      }
      if (!otherOperandSize || !otherOperandOffset)
        return std::nullopt;
      params.dimShapeSize = otherOperandSize;
      params.dimOffset = otherOperandOffset;
    }

    return params;
  }

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
  bool analyzeMaskChain(DimComparisonParams &params, Value maskVal) const {

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

    // 只有当多个mask, maskanalysis才用min限定范围
    if (auto minsiOp = maskVal.getDefiningOp<arith::MinSIOp>()) {
      bool foundTensorAxis = false;
      for (Value operand : minsiOp.getOperands()) {
        if (auto constOp = operand.getDefiningOp<arith::ConstantOp>()) {
          // Compare against the expected block dimension size.
          if (mlir::cast<IntegerAttr>(constOp.getValue()).getInt() ==
              params.blockDimSize) {
            foundTensorAxis = true;
            // Select the operand that is not the constant.
            newDim =
                minsiOp.getOperand(operand == minsiOp.getOperand(0) ? 1 : 0);
          }
        }
      }
      if (!foundTensorAxis)
        return false;
    }

    // Find the subtraction op in the chain.
    auto subiOp = newDim.getDefiningOp<arith::SubIOp>();
    if (!subiOp)
      return false;

    // Check that the final statement in the transformed mask computation
    // (i.e., "%newDim = arith.sub %newEnd2, (pid_n * BLOCK_SIZE_N)") has the
    // expected offset.
    bool hasExpectedOffset = false;
    Value newEndInSub = nullptr;
    for (Value operand : subiOp.getOperands()) {
      if (operand == params.dimOffset)
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
    bool foundOffsetInMax = false;
    Value newEnd = nullptr;
    for (Value operand : maxsiOp.getOperands()) {
      if (operand == params.dimOffset)
        foundOffsetInMax = true;
      else
        newEnd = operand;
    }
    if (!foundOffsetInMax || !newEnd)
      return false;

    // Finally, ensure that the other operand of the max originates from a min
    // operation that uses a constant equal to the make_tptr shape value.
    //" %newEnd   = arith.min %lhsEnd, N"
    auto innerMinsiOp = newEnd.getDefiningOp<arith::MinSIOp>();
    if (!innerMinsiOp)
      return false;
    bool foundShapeConst = false;
    Value lhsEnd = nullptr;
    for (Value operand : innerMinsiOp.getOperands()) {
      if (operand == params.dimShapeSize)
        foundShapeConst = true;
      else
        lhsEnd = operand;
    }
    if (!foundShapeConst)
      return false;

    // judge pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    //  addiOp->getOperand(0)  -> pid_m * BLOCK_SIZE_M
    // addiOp->getOperand(1)->  tl.arange(0, BLOCK_SIZE_M)
    auto addiOp = lhsEnd.getDefiningOp<arith::AddIOp>();
    if (!addiOp)
      return false;
    auto constOp = (addiOp->getOperand(1)).getDefiningOp<arith::ConstantOp>();
    if (constOp &&
        mlir::cast<IntegerAttr>(constOp.getValue()).getInt() == params.blockDimSize &&
        addiOp->getOperand(0) == params.dimOffset)
      return true;

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
    // Use the helper function to get the dimension parameters.
    std::optional<DimComparisonParams> paramsOpt =
        getDimComparisonParams(op, d);
    if (!paramsOpt)
      return false;

    // For debugging purposes:
    llvm::errs() << "Dimension " << d << ": dimShapeSize = ";
    paramsOpt->dimShapeSize.getDefiningOp()->dump();
    llvm::errs() << ", dimOffset = ";
    paramsOpt->dimOffset.getDefiningOp()->dump();
    llvm::errs() << ", blockDimSize = " << paramsOpt->blockDimSize << "\n";

    // Check that the op result has exactly one use.
    Value opResult = op.getResult();
    if (!opResult.hasOneUse())
      return false;

    // Ensure that the direct use is a tts.load op.
    Operation *ttsLoadOp = *opResult.user_begin();

    if (!isa<tts::LoadOp>(ttsLoadOp) || !ttsLoadOp->hasOneUse())
      return false;

    // Search for an indirect use that eventually leads to a tts.store op.
    Value storeMaskCandidate = nullptr;
    if (failed(traceUseChainToStore(ttsLoadOp->getResult(0), d,
                                    storeMaskCandidate))) {
      return false;
    }

    // Call the helper function to analyze the mask chain.
    // (See the definition of analyzeMaskChain above for details.)
    return analyzeMaskChain(*paramsOpt, storeMaskCandidate);
  };
}; // end anonymous namespace
} // namespace
/// Helper function to add the rewrite pattern to a pattern list.
void mlir::triton::populateSimplifyTTSMakeTPtrPatterns(
    RewritePatternSet &patterns) {
  patterns.add<AnalyzeAndTransformTTSMakeTPtrPattern>(patterns.getContext());
}