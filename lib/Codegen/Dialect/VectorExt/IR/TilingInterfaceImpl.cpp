#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"

using namespace mlir;
using namespace mlir::tts::IREE::VectorExt;

namespace {
namespace tts_impl {

/// Helper to get a Value from an OpFoldResult, creating a ConstantIndexOp if
/// needed.
static Value getValueOrCreateConstantIndexOp(OpBuilder &b, Location loc,
                                             OpFoldResult valueOrAttr) {
  if (auto attr = dyn_cast<Attribute>(valueOrAttr)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return b.create<arith::ConstantIndexOp>(loc, intAttr.getInt());
  }
  return cast<Value>(valueOrAttr);
}

} // namespace tts_impl

/// Implements the TilingInterface for IREEVectorExt::TransferWriteOp.
struct TransferWriteTilingInterface
    : public TilingInterface::ExternalModel<TransferWriteTilingInterface,
                                            TransferWriteOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto writeOp = cast<TransferWriteOp>(op);
    auto shapedType = cast<ShapedType>(writeOp.getValue().getType());
    return SmallVector<utils::IteratorType>(shapedType.getRank(),
                                            utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto writeOp = cast<TransferWriteOp>(op);
    auto loc = writeOp.getLoc();
    auto shapedType = cast<ShapedType>(writeOp.getValue().getType());

    SmallVector<Range> ranges;
    ranges.reserve(shapedType.getRank());
    for (int64_t i = 0; i < shapedType.getRank(); ++i) {
      Value dim =
          b.create<arith::ConstantIndexOp>(loc, shapedType.getDimSize(i));
      Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
      Value one = b.create<arith::ConstantIndexOp>(loc, 1);
      ranges.push_back(Range{zero, dim, one});
    }
    return ranges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto writeOp = cast<TransferWriteOp>(op);
    auto loc = writeOp.getLoc();

    // Compute new indices by adding offsets to original mixed indices.
    SmallVector<OpFoldResult> newIndices;
    newIndices.reserve(offsets.size());
    for (auto en : llvm::enumerate(offsets)) {
      OpFoldResult origIdx = writeOp.getMixedIndices()[en.index()];
      Value offVal =
          tts_impl::getValueOrCreateConstantIndexOp(b, loc, en.value());
      Value origVal =
          tts_impl::getValueOrCreateConstantIndexOp(b, loc, origIdx);
      Value newVal = b.create<arith::AddIOp>(loc, origVal, offVal);
      newIndices.push_back(newVal);
    }

    // Extract a subview of the original value memref.
    Value originalValue = writeOp.getValue();
    SmallVector<OpFoldResult> strides(sizes.size(), b.getIndexAttr(1));
    Value subView = b.create<memref::SubViewOp>(loc, originalValue, offsets, sizes, strides);

    // Create the tiled TransferWriteOp.
    auto newOp =
        b.create<TransferWriteOp>(loc, writeOp.getBase(), subView, newIndices,
                                  writeOp.getMixedMaskDims());

    TilingResult result;
    result.tiledOps.push_back(newOp);
    return result;
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    // No result values for TransferWriteOp.
    return success();
  }
};

} // end anonymous namespace

void mlir::tts::IREE::VectorExt::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, IREEVectorExtDialect *dialect) {
    TransferWriteOp::attachInterface<TransferWriteTilingInterface>(*ctx);
  });
}
