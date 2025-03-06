
#ifndef TRITON_SHARED_CODEGEN_DIALECT_TTS_GPU_OPS_H_
#define TRITON_SHARED_CODEGEN_DIALECT_TTS_GPU_OPS_H_

#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUAttrs.h"
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

// clang-format off
#define GET_OP_CLASSES
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUOps.h.inc"  // IWYU pragma: export
// clang-format on

#endif  // TRITON_SHARED_CODEGEN_DIALECT_TTS_GPU_OPS_H_
