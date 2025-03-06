#ifndef TRITON_SHARED_CODEGEN_DIALECT_GPU_TRANSFORMEXTENSIONS_TTSGPUEXTENSIONS_H_
#define TRITON_SHARED_CODEGEN_DIALECT_GPU_TRANSFORMEXTENSIONS_TTSGPUEXTENSIONS_H_

#include "triton-shared/Codegen/Dialect/GPU/IR/TtsGPUInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

namespace mlir {

class DialectRegistry;

namespace func {
class FuncOp;
} // namespace func

namespace linalg {
class LinalgOp;
} // namespace linalg

namespace transform {
// Types needed for builders.
class TransformTypeInterface;
} // namespace transform

} // namespace mlir

#define GET_OP_CLASSES
#include "triton-shared/Codegen/Dialect/GPU/TransformExtensions/TtsGPUExtensionsOps.h.inc"

namespace mlir::tts {

/// Registers transformations for the Triton Shared GPU dialect.
void registerTransformDialectTtsGPUExtension(DialectRegistry &registry);

namespace transform_dialect {
/// Hook to register common transformations to the transform dialect.
class TtsGPUExtensions
    : public transform::TransformDialectExtension<TtsGPUExtensions> {
public:
  TtsGPUExtensions();
};
} // namespace transform_dialect

} // namespace mlir::tts

#endif // TRITON_SHARED_CODEGEN_DIALECT_GPU_TRANSFORMEXTENSIONS_TTSGPUEXTENSIONS_H_
