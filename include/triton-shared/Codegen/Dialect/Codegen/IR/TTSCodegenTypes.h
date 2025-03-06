#ifndef TTS_SHARED_DIALECT_GPU_IR_TTS_GPU_TYPES_H_
#define TTS_SHARED_DIALECT_GPU_IR_TTS_GPU_TYPES_H_

#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::tts::GPU {
//===----------------------------------------------------------------------===//
// Layout Struct Types.
//===----------------------------------------------------------------------===//

// Metadata for a swizzle, that is, an (expand_shape -> transposition)
// pair of ops performing a change of layout within the tiles. This is used
// on GPU, where the tiles themselves can have an arbitrary layout.
struct TileSwizzle {
  struct Dim {
    // Describes what varies across this dimension.
    enum class Kind : int8_t {
      // This dimension is internal to one intrinsic on one thread. This
      // is only seen for intrinsic operands that are themselves vectors.
      Internal,
      // This dimension is internal to one intrinsic, but is across threads.
      CrossThread,
      // This dimensions is across intrinsics, as in, actual instructions in the
      // generated code.
      CrossIntrinsic
    };

    Kind kind = Kind::Internal;

    // The size of the dimension.
    int16_t size = 0;

    // Support constructing from any size type.
    template <typename T>
    Dim(Kind kind, T size) : kind(kind), size(size) {}
  };

  using ExpandShapeDimVectorType = llvm::SmallVector<Dim, 4>;
  using ExpandShapeType = llvm::SmallVector<ExpandShapeDimVectorType>;

  ExpandShapeType expandShape;
  llvm::SmallVector<int64_t> permutation;
};

/// Container of information needed to materialize the layout transformations.
struct MaterializeEncodingInfo {
  SmallVector<int64_t> innerDimsPos;
  SmallVector<int64_t> innerTileSizes;
  SmallVector<int64_t> outerDimsPerm;

  std::optional<TileSwizzle> swizzle;
};

} // namespace mlir::tts::GPU

#endif // TTS_SHARED_DIALECT_GPU_IR_TTS_GPU_TYPES_H_
