#include <memory>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"

using namespace mlir;
using namespace triton;
namespace mlir {
namespace triton {

#define GEN_PASS_DEF_REINTERPRETCASTHOIST
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

} // namespace triton
} // namespace mlir

namespace {

class ReinterpretCastHoistPass
    : public triton::impl::ReinterpretCastHoistBase<ReinterpretCastHoistPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    getOperation().walk([&](memref::ReinterpretCastOp op) {
      Operation *insertPt = nullptr;
      for (Value operand : op.getOperands()) {
        if (Operation *defOp = operand.getDefiningOp()) {
          if (!insertPt || insertPt->isBeforeInBlock(defOp))
            insertPt = defOp;
        }
      }
      if (insertPt)
        op.getOperation()->moveAfter(insertPt);
    });
  }
};
} // namespace

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createReinterpretCastHoistPass() {
  return std::make_unique<ReinterpretCastHoistPass>();
}
} // namespace triton
} // namespace mlir