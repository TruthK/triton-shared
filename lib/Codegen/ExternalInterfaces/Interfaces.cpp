// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "triton-shared/Codegen/ExternalInterfaces/Interfaces.h"

// #include "triton-shared/Codegen/ExternalInterfaces/CPUEncodingExternalModels.h"
#include "triton-shared/Codegen/ExternalInterfaces/GPUEncodingExternalModels.h"

namespace mlir::tts {

void registerCodegenExternalInterfaces(DialectRegistry &registry) {
  IREE::GPU::registerGPUEncodingExternalModels(registry);
  // IREE::CPU::registerCPUEncodingExternalModels(registry);
}

} // namespace mlir::tts
