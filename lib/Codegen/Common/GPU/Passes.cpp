// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "triton-shared/Codegen/Common/GPU/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::tts {

namespace {
#define GEN_PASS_REGISTRATION
#include "triton-shared/Codegen/Common/GPU/Passes.h.inc"
} // namespace

void registerCodegenCommonGPUPasses() {
  // Generated.
  registerPasses();
}
} // namespace mlir::tts
