#ifndef LINALG_TO_LLVM_CONVERSION_H
#define LINALG_TO_LLVM_CONVERSION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tts {

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToLLVMPass();

} // namespace triton
} // namespace mlir

#endif
