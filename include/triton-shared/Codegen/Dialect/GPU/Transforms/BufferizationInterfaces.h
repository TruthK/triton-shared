

#ifndef TTS_CODEGEN_DIALECT_GPU_TRANSFORMS_BUFFERIZATIONINTERFACES_H_
#define TTS_CODEGEN_DIALECT_GPU_TRANSFORMS_BUFFERIZATIONINTERFACES_H_

#include "mlir/IR/Dialect.h"

namespace mlir::tts {

// Register all interfaces needed for bufferization.
void registerIREEGPUBufferizationInterfaces(DialectRegistry &registry);

} // namespace mlir::tts

#endif // TTS_CODEGEN_DIALECT_GPU_TRANSFORMS_BUFFERIZATIONINTERFACES_H_
