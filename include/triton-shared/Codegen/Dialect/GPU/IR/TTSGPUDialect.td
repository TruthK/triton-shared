#ifndef TTS_CODEGEN_DIALECT_GPU_TTSGPU_DIALECT
#define TTS_CODEGEN_DIALECT_GPU_TTSGPU_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// TTS GPU Dialect
//===----------------------------------------------------------------------===//

def TTSGPU_Dialect : Dialect {
  let name = "tts_gpu";
  let cppNamespace = "::mlir::tts::GPU";

  let summary = [{
    A dialect for common functionality used by GPU focused code generation.
  }];
  let description = [{
    This dialect provides operations and attributes to aid in code generation
    for GPU targets. The functionality in this dialect can be hardware specific,
    but is intended to be independent of the lowering target. Late lowerings to
    SPIR-V/LLVM are handled separately.
  }];
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

class RankedTensorOrVectorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes,
      Or<[IsVectorOfNonZeroRankTypePred, And<[IsTensorTypePred, HasRankPred]>]>,
  "ranked tensor or vector", "::mlir::ShapedType">;

def AnyRankedTensorOrVector : RankedTensorOrVectorOf<[AnyType]>;

#endif // TTS_CODEGEN_DIALECT_GPU_TTSGPU_DIALECT
