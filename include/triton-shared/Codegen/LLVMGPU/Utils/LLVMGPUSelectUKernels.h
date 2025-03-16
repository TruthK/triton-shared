// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "triton-shared/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::tts {

IREE::GPU::UKernelConfigAttr selectUKernel(Operation *op);

} // namespace mlir::tts
