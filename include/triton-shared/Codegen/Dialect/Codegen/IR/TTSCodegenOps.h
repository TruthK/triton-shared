#ifndef TTS_CODEGEN_DIALECT_IREECODEGENOPS_H_
#define TTS_CODEGEN_DIALECT_IREECODEGENOPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// clang-format off
#define GET_OP_CLASSES
#include "triton-shared/Codegen/Dialect/Codegen/IR/TTSCodegenOps.h.inc"  // IWYU pragma: export
// clang-format on

#endif  // #ifndef TTS_CODEGEN_DIALECT_IREECODEGENOPS_H_
