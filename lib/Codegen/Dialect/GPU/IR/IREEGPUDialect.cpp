// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "triton-shared/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"

#include "triton-shared/Codegen/Dialect/GPU/IR/IREEGPUDialect.cpp.inc"
#include "triton-shared/Codegen/Dialect/GPU/IR/IREEGPUOps.h"

namespace mlir::tts::IREE::GPU {

void IREEGPUDialect::initialize() {
  registerAttributes();

  addOperations<
#define GET_OP_LIST
#include "triton-shared/Codegen/Dialect/GPU/IR/IREEGPUOps.cpp.inc"
      >();
}

} // namespace mlir::tts::IREE::GPU
