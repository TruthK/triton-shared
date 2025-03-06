#ifndef TTS_COMPILER_CODEGEN_DIALECT_GPU_TTSGPUDIALECT_H_
#define TTS_COMPILER_CODEGEN_DIALECT_GPU_TTSGPUDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

// clang-format off: must be included after all LLVM/MLIR headers
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUDialect.h.inc" // IWYU pragma: keep
// clang-format on

#endif // TTS_COMPILER_CODEGEN_DIALECT_GPU_TTSGPUDIALECT_H_
