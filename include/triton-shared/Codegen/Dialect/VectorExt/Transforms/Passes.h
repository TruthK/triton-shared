// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_PASSES_H_

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tts::IREE::VectorExt {
#define GEN_PASS_DECL
#include "triton-shared/Codegen/Dialect/VectorExt/Transforms/Passes.h.inc" // IWYU pragma: keep
} // namespace mlir::tts::IREE::VectorExt

namespace mlir::tts {
/// Register VectorExt passes.
void registerIREEVectorExtPasses();
} // namespace mlir::tts

#endif // IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_PASSES_H_
