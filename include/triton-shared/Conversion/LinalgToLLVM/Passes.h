#ifndef TRITON_TO_LINALG_TO_LLVM_CONVERSION_PASSES_H
#define TRITON_TO_LINALG_TO_LLVM_CONVERSION_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace tts {
// struct LinalgToLLVMOptions : public PassPipelineOptions<LinalgToLLVMOptions> {
//   PassOptions::Option<bool> reassociateFPReductions{
//       *this, "reassociate-fp-reductions",
//       llvm::cl::desc("Allow reassociation og FP reductions"),
//       llvm::cl::init(false)};
// };

void registerLinalgToLLVM();

void buildLinalgToLLVMPipelinePass(OpPassManager &pm);

// #define GEN_PASS_REGISTRATION
// #include "triton-shared/Conversion/LinalgToLLVM/Passes.h.inc"

} // namespace tts
} // namespace mlir

#endif
