#include "triton-shared/Codegen/Dialect/GPU/IR/DerivedConfigUtils.h"
#include <numeric>

#include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenAttrs.h"
#include "triton-shared/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::tts::GPU {

static constexpr int64_t kPreferredCopyNumBits = 128;

static SmallVector<int64_t>
getVectorSizeTileSizes(int64_t rank, int64_t innerDimSize, int64_t vectorSize) {
  SmallVector<int64_t> tileSizes(rank, 1);
  if (ShapedType::isDynamic(innerDimSize) || innerDimSize >= vectorSize) {
    tileSizes.back() = vectorSize;
  } else {
    tileSizes.back() = innerDimSize;
  }
  return tileSizes;
}

static SmallVector<int64_t>
getVectorTileSizesFromLoopRanges(SmallVector<int64_t> loopRanges,
                                 int64_t numThreads, int64_t vectorSize,
                                 bool allowMultiDimCollapse = true) {
  if (llvm::any_of(loopRanges, &ShapedType::isDynamic)) {
    return getVectorSizeTileSizes(loopRanges.size(), loopRanges.back(),
                                  vectorSize);
  }

  int64_t flatNumTrips = std::accumulate(loopRanges.begin(), loopRanges.end(),
                                         1, std::multiplies<int64_t>());
  if (flatNumTrips % numThreads != 0) {
    return getVectorSizeTileSizes(loopRanges.size(), loopRanges.back(),
                                  vectorSize);
  }
  SmallVector<int64_t> tileSizes(loopRanges.size(), 1);

  int64_t maxVectorSize = std::min(vectorSize, flatNumTrips / numThreads);

  int64_t innerMostRange = loopRanges.back();
  if (innerMostRange % maxVectorSize != 0 &&
      maxVectorSize % innerMostRange != 0) {
    return tileSizes;
  }

  tileSizes.back() = std::min(innerMostRange, maxVectorSize);
  if (innerMostRange >= maxVectorSize || !allowMultiDimCollapse) {
    return tileSizes;
  }

  maxVectorSize = maxVectorSize / innerMostRange;
  for (int64_t i = loopRanges.size() - 2, e = 0; i >= e; --i) {
    int64_t range = loopRanges[i];
    if (maxVectorSize % range != 0) {
      break;
    }
    tileSizes[i] = range;
    maxVectorSize = maxVectorSize / range;
  }

  return tileSizes;
}

SmallVector<int64_t> deriveLinalgOpThreadTileSizes(linalg::LinalgOp linalgOp,
                                                   int64_t numThreads) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return {};
  }
  SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
  int64_t vectorSize = kPreferredCopyNumBits /
                       getElementTypeOrSelf(linalgOp->getResultTypes()[0])
                           .getIntOrFloatBitWidth();
  SmallVector<int64_t> tileSizes =
      getVectorTileSizesFromLoopRanges(loopRanges, numThreads, vectorSize);
  for (auto [tileSize, iterType] :
       llvm::zip(tileSizes, linalgOp.getIteratorTypesArray())) {
    if (iterType == utils::IteratorType::reduction) {
      tileSize = 0;
    }
  }
  return tileSizes;
}


SmallVector<int64_t> deriveThreadTileSizes(Operation *op) {
  std::optional<SmallVector<int64_t>> workgroupSize =
      getWorkgroupSize(op->getParentOfType<FunctionOpInterface>());
  if (!workgroupSize) {
    return {};
  }
  int64_t numThreads =
      std::accumulate(workgroupSize->begin(), workgroupSize->end(), 1,
                      std::multiplies<int64_t>());
  return TypeSwitch<Operation *, SmallVector<int64_t>>(op)
      .Case([&](linalg::LinalgOp linalgOp) -> SmallVector<int64_t> {
        return deriveLinalgOpThreadTileSizes(linalgOp, numThreads);
      })
      .Default([](Operation *op) -> SmallVector<int64_t> { return {}; });
}

} // namespace mlir::tts::GPU
