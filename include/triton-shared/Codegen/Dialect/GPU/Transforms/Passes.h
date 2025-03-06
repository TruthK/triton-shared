


#ifndef TTS_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES_H_
#define TTS_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES_H_

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tts::GPU {
#define GEN_PASS_DECL
#include "triton-shared/Codegen/Dialect/GPU/Transforms/Passes.h.inc" // IWYU pragma: keep
} // namespace mlir::tts::GPU

namespace mlir::tts {
/// Register GPU passes.
void registerIREEGPUPasses();
} // namespace mlir::tts

#endif // TTS_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES_H_
