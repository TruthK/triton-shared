

#ifndef TTS_SHARED_CODEGEN_DIALECT_UKERNELOPS_H_
#define TTS_SHARED_CODEGEN_DIALECT_UKERNELOPS_H_

#include "triton-shared/Codegen/Interfaces/UKernelOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

// clang-format off
#define GET_OP_CLASSES
#include "triton-shared/Codegen/Dialect/Codegen/IR/UKernelOps.h.inc"  // IWYU pragma: export
// clang-format on

#endif  // TTS_SHARED_CODEGEN_DIALECT_UKERNELOPS_H_