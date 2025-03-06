
#ifndef TRITON_SHARED_CODEGEN_DIALECT_IREECODEGEN_DIALECT_H_
#define TRITON_SHARED_CODEGEN_DIALECT_IREECODEGEN_DIALECT_H_

#include <mutex>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

// clang-format off: must be included after all LLVM/MLIR headers
#include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenDialect.h.inc"  // IWYU pragma: keep
// clang-format on

namespace mlir::tts {

void registerUKernelBufferizationInterface(DialectRegistry &registry);

}  // namespace mlir::tts

#endif  // TRITON_SHARED_CODEGEN_DIALECT_IREECODEGEN_DIALECT_H_
