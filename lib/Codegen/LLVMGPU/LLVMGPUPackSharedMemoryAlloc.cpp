// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "triton-shared/Codegen/LLVMGPU/Passes.h"
#include "triton-shared/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"

namespace mlir::tts {

#define GEN_PASS_DEF_LLVMGPUPACKSHAREDMEMORYALLOCPASS
#include "triton-shared/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct LLVMGPUPackSharedMemoryAllocPass final
    : impl::LLVMGPUPackSharedMemoryAllocPassBase<
          LLVMGPUPackSharedMemoryAllocPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<nvgpu::NVGPUDialect>();
  }

  void runOnOperation() override {
    packSharedMemoryAlloc(getOperation());
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUPackSharedMemoryAlloc() {
  return std::make_unique<LLVMGPUPackSharedMemoryAllocPass>();
}

} // namespace mlir::tts
