#ifndef TRITON_TO_LINALG_TO_LLVM_CONVERSION_PASSES_H
#define TRITON_TO_LINALG_TO_LLVM_CONVERSION_PASSES_H

#include "llvm/ADT/StringMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace tts {
void registerLinalgToLLVM();

void buildLinalgToLLVMPipelinePass(OpPassManager &pm);

// #define GEN_PASS_REGISTRATION
// #include "triton-shared/Conversion/LinalgToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
