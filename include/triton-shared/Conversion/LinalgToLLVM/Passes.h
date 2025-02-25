#ifndef LINALG_TO_LLVM_CONVERSION_PASSES_H
#define LINALG_TO_LLVM_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tts {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "triton-shared/Conversion/LinalgToLLVM/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/LinalgToLLVM/Passes.h.inc"


} // namespace tts
} // namespace mlir

#endif
