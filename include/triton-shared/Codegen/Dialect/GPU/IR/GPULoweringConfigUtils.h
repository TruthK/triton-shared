#ifndef TTS_COMPILER_CODEGEN_DIALECT_GPU_IR_GPULOWERINGCONFIGUTILS_H_
#define TTS_COMPILER_CODEGEN_DIALECT_GPU_IR_GPULOWERINGCONFIGUTILS_H_

#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUAttrs.h"

namespace mlir::tts::GPU {

/// Helper to retrieve/set a target mma intrinsic.
MmaInterfaceAttr getMmaKind(LoweringConfigAttr config);
void setMmaKind(MLIRContext *context, SmallVectorImpl<NamedAttribute> &attrs,
                MmaInterfaceAttr kind);

// TODO: Merge subgroup counts functionality into subgroup tiling level
//       lowering, when we have it implemented.
/// Helper to retrieve/set a target subgroup M/N counts.
std::optional<int64_t> getSubgroupMCount(LoweringConfigAttr config);
std::optional<int64_t> getSubgroupNCount(LoweringConfigAttr config);
void setSubgroupMCount(MLIRContext *context,
                       SmallVectorImpl<NamedAttribute> &attrs,
                       int64_t subgroupMCount);
void setSubgroupNCount(MLIRContext *context,
                       SmallVectorImpl<NamedAttribute> &attrs,
                       int64_t subgroupNCount);

// The basis consists of two integer arrays:
//   - "counts": number of resource to use per dimension in the basis.
//   - "mapping": a projected permutation to map to basis to the operations
//     iteration space.
//
// Given a resource "x", the "basis" can be used to determine the distribution
// of an iteration space using:
//
// b = delinearize(x, counts)
// idx = apply(b, mapping)
struct Basis {
  SmallVector<int64_t> counts;
  SmallVector<int64_t> mapping;
};

// Helper to retrieve/set distribution basis.
FailureOr<Basis> getBasis(tts::GPU::LoweringConfigAttr config,
                          tts::GPU::TilingLevel level);
void setBasis(MLIRContext *context, SmallVector<NamedAttribute> &attrs,
              tts::GPU::TilingLevel level, const Basis &basis);

/// Helper to retrieve/set a list of operand indices to promote.
std::optional<SmallVector<int64_t>>
getPromotedOperandList(LoweringConfigAttr config);
void setPromotedOperandList(MLIRContext *context,
                            SmallVectorImpl<NamedAttribute> &attrs,
                            ArrayRef<int64_t> operands);

/// Helper to retrieve  list of operand to pad.
std::optional<SmallVector<int64_t>> getPaddingList(LoweringConfigAttr config);

tts::GPU::UKernelConfigAttr
getUkernelSpec(tts::GPU::LoweringConfigAttr config);

} // namespace mlir::tts::GPU

#endif // TTS_COMPILER_CODEGEN_DIALECT_GPU_IR_GPULOWERINGCONFIGUTILS_H_
