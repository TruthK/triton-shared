#include "triton-shared/Codegen/Dialect/GPU/IR/GPUTileSwizzleUtils.h"
#include "triton-shared/Codegen/Dialect/Codegen/Utils/Utils.h"

namespace mlir::tts::GPU {

using ::mlir::tts::Codegen::TileSwizzle;
using Kind = TileSwizzle::Dim::Kind;

static int64_t expandedDimIdx(const TileSwizzle::ExpandShapeType &expandShape,
                              int srcIdx) {
  int dstIdx = 0;
  for (int i = 0; i < srcIdx; ++i) {
    dstIdx += expandShape[i].size();
  }
  return dstIdx;
}

static void expand(TileSwizzle &swizzle, int srcIdx, TileSwizzle::Dim dim) {
  int dstIdx = expandedDimIdx(swizzle.expandShape, srcIdx);
  swizzle.expandShape[srcIdx].insert(swizzle.expandShape[srcIdx].begin(), dim);
  for (auto &p : swizzle.permutation) {
    p += (p >= dstIdx);
  }
  swizzle.permutation.insert(swizzle.permutation.begin(), dstIdx);
}

static void interleave(TileSwizzle &swizzle, int srcIdx, int expandedIdx) {
  int dstIdx = expandedDimIdx(swizzle.expandShape, srcIdx) + expandedIdx;
  SmallVector<int64_t> outPermutation(swizzle.permutation.size());
  outPermutation[dstIdx] = swizzle.permutation[0];
  for (int i = 0; i < dstIdx; ++i) {
    outPermutation[i] = swizzle.permutation[i + 1];
  }
  for (int i = dstIdx + 1; i < outPermutation.size(); ++i) {
    outPermutation[i] = swizzle.permutation[i];
  }
  swizzle.permutation = outPermutation;
}

TileSwizzle getIntrinsicSwizzle(tts::GPU::MMAIntrinsic intrinsic,
                                tts::GPU::MMAFragment fragment) {
  auto layout = tts::GPU::getSingleSubgroupLayout(intrinsic, fragment);

  if (fragment == tts::GPU::MMAFragment::Rhs) {
    std::swap(layout.outer[0], layout.outer[1]);
    std::swap(layout.thread[0], layout.thread[1]);
    std::swap(layout.tstrides[0], layout.tstrides[1]);
    std::swap(layout.element[0], layout.element[1]);
  }

  TileSwizzle swizzle;
  assert(layout.thread.size() == 2);
  swizzle.expandShape.resize(2);
  for (auto [i, e] : llvm::enumerate(layout.element)) {
    if (e != 1) {
      expand(swizzle, i, {Kind::Internal, e});
    }
  }
  for (auto [i, t] : llvm::enumerate(layout.thread)) {
    if (t != 1) {
      expand(swizzle, i, {Kind::CrossThread, t});
    }
  }
  if (layout.thread[0] != 1 && layout.thread[1] != 1 &&
      layout.tstrides[0] > layout.tstrides[1]) {
    std::swap(swizzle.permutation[0], swizzle.permutation[1]);
  }
  for (auto [i, o] : llvm::enumerate(layout.outer)) {
    if (o != 1) {
      expand(swizzle, i, {Kind::Internal, o});
    }
  }
  return swizzle;
}

static int getInnermostNonInternalDimIdx(
    const TileSwizzle::ExpandShapeDimVectorType &shape) {
  for (int idx = shape.size() - 1; idx >= 0; --idx) {
    if (shape[idx].kind != Kind::Internal) {
      return idx;
    }
  }
  assert(false && "all dimensions are internal!");
  return 0;
}

TileSwizzle getSwizzle(tts::GPU::DataTiledMMAAttr mma,
                       tts::GPU::MMAFragment fragment) {
  auto swizzle = getIntrinsicSwizzle(mma.getIntrinsic().getValue(), fragment);
  switch (fragment) {
  case tts::GPU::MMAFragment::Lhs:
    if (mma.getUnrollK() > 1) {
      expand(swizzle, 1, {Kind::CrossIntrinsic, mma.getUnrollK()});
      int interleavingIdx =
          getInnermostNonInternalDimIdx(swizzle.expandShape[1]);
      interleave(swizzle, 1, interleavingIdx);
    }
    if (mma.getUnrollM() > 1) {
      expand(swizzle, 0, {Kind::CrossIntrinsic, mma.getUnrollM()});
    }
    if (mma.getSubgroupsM() > 1) {
      expand(swizzle, 0, {Kind::CrossThread, mma.getSubgroupsM()});
    }
    break;
  case tts::GPU::MMAFragment::Rhs:
    if (mma.getUnrollK() > 1) {
      expand(swizzle, 1, {Kind::CrossIntrinsic, mma.getUnrollK()});
      int interleavingIdx =
          getInnermostNonInternalDimIdx(swizzle.expandShape[1]);
      interleave(swizzle, 1, interleavingIdx);
    }
    if (mma.getUnrollN() > 1) {
      expand(swizzle, 0, {Kind::CrossIntrinsic, mma.getUnrollN()});
    }
    if (mma.getSubgroupsN() > 1) {
      expand(swizzle, 0, {Kind::CrossThread, mma.getSubgroupsN()});
    }
    break;
  case tts::GPU::MMAFragment::Acc:
    if (mma.getUnrollN() > 1) {
      expand(swizzle, 1, {Kind::CrossIntrinsic, mma.getUnrollN()});
    }
    if (mma.getUnrollM() > 1) {
      expand(swizzle, 0, {Kind::CrossIntrinsic, mma.getUnrollM()});
    }
    if (mma.getSubgroupsN() > 1) {
      expand(swizzle, 1, {Kind::CrossThread, mma.getSubgroupsN()});
    }
    if (mma.getSubgroupsM() > 1) {
      expand(swizzle, 0, {Kind::CrossThread, mma.getSubgroupsM()});
    }
    break;
  }
  return swizzle;
}

} // namespace mlir::tts::GPU
