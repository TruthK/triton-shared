#include <memory>

#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"

using namespace mlir;
using namespace triton;
namespace mlir {
namespace triton {

#define GEN_PASS_DEF_TTSLOOPUNROLL
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

} // namespace triton
} // namespace mlir

namespace {

class TTSLoopUnroll : public triton::impl::TTSLoopUnrollBase<TTSLoopUnroll> {

  int getUnrollFactorOrDefault(scf::ForOp forOp) {
    // 获取循环的上下界和步长
    Value lowerBound = forOp.getLowerBound();
    Value upperBound = forOp.getUpperBound();
    Value step = forOp.getStep();

    // 尝试提取常量值
    APInt lbAPInt, ubAPInt, stepAPInt;
    if (matchPattern(lowerBound, m_ConstantInt(&lbAPInt)) &&
        matchPattern(upperBound, m_ConstantInt(&ubAPInt)) &&
        matchPattern(step, m_ConstantInt(&stepAPInt))) {
      // 转换为int64_t进行计算
      int64_t lbValue = lbAPInt.getSExtValue();
      int64_t ubValue = ubAPInt.getSExtValue();
      int64_t stepValue = stepAPInt.getSExtValue();

      if (stepValue != 0) {
        // 计算需要的循环次数作为展开因子
        int64_t tripCount = (ubValue - lbValue + stepValue - 1) / stepValue;
        return tripCount;
      }
    }

    return 1;
  }

public:
  void runOnOperation() override {
    SmallVector<scf::ForOp, 4> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    auto ctx = getOperation()->getContext();
    for (auto loop : loops) {
      auto unrollFactor = getUnrollFactorOrDefault(loop);
      if (unrollFactor > 16)
        return;

      auto resultLoops = loopUnrollByFactor(loop, unrollFactor);
      // Do not pipeline the epilog loop.
      if (succeeded(resultLoops) && resultLoops->epilogueLoopOp) {
      }
    }
  }
};
} // anonymous namespace

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createTTSLoopUnrollPass() {
  return std::make_unique<TTSLoopUnroll>();
}
} // namespace triton
} // namespace mlir
