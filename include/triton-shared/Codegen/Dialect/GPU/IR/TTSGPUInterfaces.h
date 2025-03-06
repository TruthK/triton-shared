#ifndef TRITON_SHARED_CODEGEN_DIALECT_GPU_TTSGPUINTERFACES_H_
#define TRITON_SHARED_CODEGEN_DIALECT_GPU_TTSGPUINTERFACES_H_

#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUEnums.h"
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

// clang-format off
#define GET_ATTRDEF_CLASSES
#include "triton-shared/Codegen/Dialect/GPU/IR/TTSGPUInterfaces.h.inc"
// clang-format on

#endif // TRITON_SHARED_CODEGEN_DIALECT_GPU_TTSGPUINTERFACES_H_
