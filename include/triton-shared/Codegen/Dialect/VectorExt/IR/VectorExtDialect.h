// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_VECTOREXT_IR_VECTOREXTDIALECT_H_
#define IREE_DIALECTS_DIALECT_VECTOREXT_IR_VECTOREXTDIALECT_H_

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtInterfaces.h"
// clang-format off: must be included after all LLVM/MLIR headers

namespace mlir::tts::IREE::VectorExt {
void registerTilingInterfaceExternalModels(DialectRegistry &registry);
} // namespace mlir::tts::IREE::VectorExt

#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h.inc" // IWYU pragma: keep
                                                                             //
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtEnums.h.inc" // IWYU pragma: keep

#define GET_ATTRDEF_CLASSES
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtAttrs.h.inc" // IWYU pragma: export

#define GET_OP_CLASSES
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtOps.h.inc" // IWYU pragma: export

// clang-format on

#endif // IREE_DIALECTS_DIALECT_VECTOREXT_IR_VECTOREXTDIALECT_H_
