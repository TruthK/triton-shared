#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "triton-shared/Codegen/LLVMGPU/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include <optional>

using namespace mlir;

namespace mlir::tts {

#define GEN_PASS_DEF_FUSEFORALLPASS
#include "triton-shared/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// Selects a lowering strategy for taking a hal.executable.variant operation
/// to scalar/native-vector code.
class FuseForallPass final : public impl::FuseForallPassBase<FuseForallPass> {
public:
  using impl::FuseForallPassBase<FuseForallPass>::FuseForallPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect, mlir::scf::SCFDialect,
                    mlir::tts::IREE::VectorExt::IREEVectorExtDialect>();
  }

  void runOnOperation() override;
};

// 检查两个 OpFoldResult 是否相同
static bool equalOpFoldResult(mlir::OpFoldResult lhs, mlir::OpFoldResult rhs) {
  if (auto aAttr = lhs.dyn_cast<mlir::Attribute>()) {
    if (auto bAttr = rhs.dyn_cast<mlir::Attribute>())
      return aAttr == bAttr;
    return false;
  }
  if (auto aVal = lhs.dyn_cast<mlir::Value>()) {
    if (auto bVal = rhs.dyn_cast<mlir::Value>())
      return aVal == bVal;
    return false;
  }
  return false;
}

// 判断两个 forall 的 bounds 和 step 是否完全相同
static bool haveSameBoundsAndStep(mlir::scf::ForallOp a,
                                  mlir::scf::ForallOp b) {
  auto aLB = a.getMixedLowerBound();
  auto bLB = b.getMixedLowerBound();
  auto aUB = a.getMixedUpperBound();
  auto bUB = b.getMixedUpperBound();
  auto aStep = a.getMixedStep();
  auto bStep = b.getMixedStep();
  if (aLB.size() != bLB.size() || aUB.size() != bUB.size() ||
      aStep.size() != bStep.size())
    return false;
  for (size_t i = 0; i < aLB.size(); ++i) {
    if (!equalOpFoldResult(aLB[i], bLB[i]) ||
        !equalOpFoldResult(aUB[i], bUB[i]) ||
        !equalOpFoldResult(aStep[i], bStep[i]))
      return false;
  }
  return true;
}

// 检查第二个 forall 是否使用了第一个 forall 中的值
static bool hasDependency(mlir::scf::ForallOp first,
                          mlir::scf::ForallOp second) {
  llvm::SmallPtrSet<mlir::Value, 8> defs;
  for (auto &op : first.getBody()->getOperations()) {
    for (auto result : op.getResults())
      defs.insert(result);
  }
  for (auto &op : second.getBody()->getOperations()) {
    for (auto operand : op.getOperands()) {
      if (defs.contains(operand))
        return true;
    }
  }
  return false;
}

// 模式：融合两个相邻的 scf.forall，无返回值且 bounds 一致
struct FuseAdjacentForallPattern
    : public mlir::OpRewritePattern<mlir::scf::ForallOp> {
  using mlir::OpRewritePattern<mlir::scf::ForallOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForallOp firstLoop,
                  mlir::PatternRewriter &rewriter) const override {
    auto *nextOp = firstLoop->getNextNode();
    auto secondLoop = mlir::dyn_cast_or_null<mlir::scf::ForallOp>(nextOp);
    if (!secondLoop)
      return mlir::failure();
    // 两个 forall 都不能有返回值
    if (firstLoop.getNumResults() != 0 || secondLoop.getNumResults() != 0)
      return mlir::failure();
    // bounds 和 step 必须一致
    if (!haveSameBoundsAndStep(firstLoop, secondLoop))
      return mlir::failure();
    // 确保无依赖冲突
    if (hasDependency(firstLoop, secondLoop))
      return mlir::failure();
    // 创建融合后的 forall
    auto loc = firstLoop.getLoc();
    auto lbs = firstLoop.getMixedLowerBound();
    auto ubs = firstLoop.getMixedUpperBound();
    auto steps = firstLoop.getMixedStep();
    // 构造新的
    // forall，传入bounds/step，无outputs，默认mapping，并提供空bodyBuilderFn
    auto newLoop = rewriter.create<mlir::scf::ForallOp>(
        loc, lbs, ubs, steps, mlir::ValueRange{},
        /*mapping=*/firstLoop.getMapping(),
        /*bodyBuilderFn=*/
        [](OpBuilder &builder, Location bodyLoc, ValueRange args) {});
    // // 移除自动创建的默认终结符（scf.forall.in_parallel）
    // if (auto *oldTerm = newLoop.getBody()->getTerminator())
    //   rewriter.eraseOp(oldTerm);
    // 准备映射
    auto *firstBody = firstLoop.getBody();
    auto *secondBody = secondLoop.getBody();
    auto *newBody = newLoop.getBody();
    mlir::IRMapping mapping;
    for (unsigned i = 0; i < firstBody->getNumArguments(); ++i)
      mapping.map(firstBody->getArgument(i), newBody->getArgument(i));
    // 将secondBody的block参数映射到newBody，以正确克隆第二个循环体的操作
    for (unsigned i = 0; i < secondBody->getNumArguments(); ++i)
      mapping.map(secondBody->getArgument(i), newBody->getArgument(i));
    // 克隆第一个 forall 的主体操作
    rewriter.setInsertionPointToStart(newBody);
    llvm::SmallVector<mlir::Operation *, 8> toClone;
    for (auto &op : firstBody->getOperations()) {
      if (&op == firstBody->getTerminator())
        break;
      toClone.push_back(&op);
    }
    for (auto *op : toClone)
      rewriter.clone(*op, mapping);
    // 克隆第二个 forall 的主体操作
    toClone.clear();
    for (auto &op : secondBody->getOperations()) {
      if (&op == secondBody->getTerminator())
        break;
      toClone.push_back(&op);
    }
    for (auto *op : toClone)
      rewriter.clone(*op, mapping);
    // 在新循环主体末尾插入并行循环的终结符 (scf.forall.in_parallel)
    rewriter.setInsertionPointToEnd(newBody);
    rewriter.create<mlir::scf::InParallelOp>(loc);
    // 删除原始 forall
    rewriter.eraseOp(secondLoop);
    rewriter.eraseOp(firstLoop);
    return mlir::success();
  }
};

} // namespace

void FuseForallPass::runOnOperation() {
  auto *context = &getContext();
  auto op = getOperation();
  mlir::RewritePatternSet patterns(context);
  patterns.add<FuseAdjacentForallPattern>(context);
  if (mlir::failed(mlir::applyPatternsGreedily(op, std::move(patterns))))
    signalPassFailure();
}

} // namespace mlir::tts
