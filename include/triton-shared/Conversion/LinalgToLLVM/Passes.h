#ifndef LINALG_TO_LLVM_CONVERSION_PASSES_H
#define LINALG_TO_LLVM_CONVERSION_PASSES_H

#include "triton-shared/Conversion/LinalgToLLVM/LinalgToLLVM.h"

namespace mlir {
namespace tts {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/LinalgToLLVM/Passes.h.inc"

} // namespace tts
} // namespace mlir

#endif
