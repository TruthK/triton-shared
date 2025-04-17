//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_LINALGGENERICFUSIONPASS
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

} // namespace triton
} // namespace mlir

namespace {

// 实现Pass
struct LinalgGenericFusionPass
    : public triton::impl::LinalgGenericFusionPassBase<
          LinalgGenericFusionPass> {
  void runOnOperation() final {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    auto controlFn = [](OpOperand *operand) { return true; };
    mlir::linalg::populateElementwiseOpsFusionPatterns(patterns, controlFn);

    // 为每个函数应用融合模式
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir {
namespace triton {

// 实现创建pass的工厂函数
std::unique_ptr<OperationPass<ModuleOp>> createLinalgGenericFusionPass() {
  return std::make_unique<LinalgGenericFusionPass>();
}

} // namespace triton
} // namespace mlir