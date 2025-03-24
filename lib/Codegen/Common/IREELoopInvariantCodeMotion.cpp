// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "triton-shared/Codegen/Common/Passes.h"
#include "triton-shared/Codegen/Transforms/Transforms.h"

namespace mlir::tts {

#define GEN_PASS_DEF_IREELOOPINVARIANTCODEMOTIONPASS
#include "triton-shared/Codegen/Common/Passes.h.inc"

namespace {
/// IREE loop invariant code motion (LICM) pass.
struct IREELoopInvariantCodeMotionPass
    : public impl::IREELoopInvariantCodeMotionPassBase<
          IREELoopInvariantCodeMotionPass> {
  void runOnOperation() override;
};
} // namespace

void IREELoopInvariantCodeMotionPass::runOnOperation() {
  moveLoopInvariantCodeFromGuaranteedLoops(getOperation());
}

} // namespace mlir::tts
