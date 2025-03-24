//===- TensorEmptyLoopHoisting.cpp - Hoist tensor.empty from loops -------===//
//
// This pass hoists tensor.empty operations out of loop bodies when possible.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_HOISTTENSOREMPTYFROMLOOPS
#include "triton-shared/Conversion/StructuredToMemref/Passes.h.inc"
} // namespace triton
} // namespace mlir
namespace {

/// Pass 实现：将循环中满足条件的 tensor.empty 操作提升到循环外部。
struct HoistTensorEmptyFromLoopsPass
    : public mlir::triton::impl::HoistTensorEmptyFromLoopsBase<
          HoistTensorEmptyFromLoopsPass> {
  void runOnOperation() override {
    // 获取当前函数操作
    auto func = getOperation();

    // 阶段1：收集所有处于循环内部且操作数均为循环不变式的 tensor.empty 操作
    SmallVector<tensor::EmptyOp, 8> emptyOps;
    func.walk([&](tensor::EmptyOp emptyOp) {
      // 查找最近的循环（使用 LoopLikeOpInterface 判断是否在循环中）
      auto loop = emptyOp->getParentOfType<LoopLikeOpInterface>();
      if (!loop)
        return;
      // 检查 tensor.empty 的所有操作数是否均在循环外部定义
      if (llvm::all_of(emptyOp->getOperands(), [&](Value operand) {
            return loop.isDefinedOutsideOfLoop(operand);
          }))
        emptyOps.push_back(emptyOp);
    });

    // 阶段2：将每个符合条件的 tensor.empty 操作提升到循环之前
    for (tensor::EmptyOp emptyOp : emptyOps) {
      // 再次确认操作所在的循环（因为在提升过程中可能发生变化）
      auto loop = emptyOp->getParentOfType<LoopLikeOpInterface>();
      if (!loop)
        continue;
      // 将 tensor.empty 操作移动到循环之前，从而使得其仅在循环外部创建一次
      emptyOp->moveBefore(loop);
    }
  }
};

} // end anonymous namespace
/// Create the pass instance.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::triton::createHoistTensorEmptyFromLoopsPass() {
  return std::make_unique<HoistTensorEmptyFromLoopsPass>();
}
