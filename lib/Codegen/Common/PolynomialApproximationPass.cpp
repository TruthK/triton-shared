// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Math/Transforms/Approximation.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Codegen/Common/Passes.h"

namespace mlir::tts {

/// Command line to use native hardware operations instead of polynomial
/// approximation.
static bool clNativeMathPrecision = false;

#define GEN_PASS_DEF_POLYNOMIALAPPROXIMATIONPASS
#include "triton-shared/Codegen/Common/Passes.h.inc"

namespace {

/// math dialect elementry functions -> polynomial form.
class PolynomialApproximationPass final
    : public impl::PolynomialApproximationPassBase<
          PolynomialApproximationPass> {
  void runOnOperation() override {
    RewritePatternSet mathPatterns(&getContext());
    populateExpandTanPattern(mathPatterns);
    populateExpandSinhPattern(mathPatterns);
    populateExpandCoshPattern(mathPatterns);
    populateExpandAsinhPattern(mathPatterns);
    populateExpandAcoshPattern(mathPatterns);
    populateExpandAtanhPattern(mathPatterns);
    populateExpandPowFPattern(mathPatterns);
    populateExpandFPowIPattern(mathPatterns);

    if (clNativeMathPrecision) {
      mathPatterns.add<math::ErfPolynomialApproximation>(&getContext());
    } else {
      populateExpandExp2FPattern(mathPatterns);
      populateMathPolynomialApproximationPatterns(mathPatterns);
      populateExpandRoundEvenPattern(mathPatterns);
    }
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(mathPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tts
